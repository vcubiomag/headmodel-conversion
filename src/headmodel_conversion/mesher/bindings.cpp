// CGAL surface decimation + OpenCASCADE STEP B-rep export.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/LindstromTurk_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_cost.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Bounded_normal_change_filter.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Constrained_placement.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/property_map.h>
#include <memory>

#include <gp_Pnt.hxx>
#include <gp_Dir.hxx>
#include <Geom_Plane.hxx>
#include <Precision.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Solid.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Wire.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Vertex.hxx>
#include <BRep_Builder.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <STEPControl_Writer.hxx>
#include <STEPControl_StepModelType.hxx>
#include <Interface_Static.hxx>
#include <Standard_Failure.hxx>

namespace nb = nanobind;
namespace SMS = CGAL::Surface_mesh_simplification;
namespace PMP = CGAL::Polygon_mesh_processing;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
using edge_descriptor = boost::graph_traits<Mesh>::edge_descriptor;

template <typename T>
using np_in = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

// Read-only property map: true for the patch's border (feature-curve) edges.
struct Border_map {
  const Mesh* mesh;
  using key_type = edge_descriptor;
  using value_type = bool;
  using reference = bool;
  using category = boost::readable_property_map_tag;
  friend bool get(const Border_map& m, edge_descriptor e) {
    return CGAL::is_border(e, *m.mesh);
  }
};

// Spatial target-edge-length field: nearest per-vertex target over the patch's
// original vertices. Lets decimation coarsen adaptively (keep fine near the
// coil sites, coarse elsewhere) instead of to one uniform size.
class Sizing {
  using PointIdx = std::pair<Point_3, std::size_t>;
  using BaseTraits = CGAL::Search_traits_3<K>;
  using Traits = CGAL::Search_traits_adapter<
      PointIdx, CGAL::First_of_pair_property_map<PointIdx>, BaseTraits>;
  using KNN = CGAL::Orthogonal_k_neighbor_search<Traits>;
  using Tree = KNN::Tree;
  std::vector<double> targets_;
  std::shared_ptr<Tree> tree_;

public:
  Sizing(const std::vector<Point_3>& pts, const double* targets) {
    targets_.assign(targets, targets + pts.size());
    std::vector<PointIdx> data;
    data.reserve(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) data.emplace_back(pts[i], i);
    tree_ = std::make_shared<Tree>(data.begin(), data.end());
    tree_->build();
  }
  double operator()(const Point_3& p) const {
    KNN search(*tree_, p, 1);
    return targets_[search.begin()->first.second];
  }
};

// Placement that forbids collapsing an edge already at/above the local target
// edge length, so decimation stops locally when it reaches the size field.
struct Adaptive_placement {
  SMS::LindstromTurk_placement<Mesh> base;
  const Sizing* sizing;
  template <typename Profile>
  std::optional<typename Profile::Point> operator()(const Profile& p) const {
    const double tgt = (*sizing)(CGAL::midpoint(p.p0(), p.p1()));
    if (CGAL::squared_distance(p.p0(), p.p1()) >= tgt * tgt)
      return std::nullopt;
    return base(p);
  }
};

template <typename T>
static nb::ndarray<nb::numpy, T, nb::ndim<2>> own2d(T* p, size_t rows, size_t cols) {
  nb::capsule owner(p, [](void* q) noexcept { delete[] static_cast<T*>(q); });
  return nb::ndarray<nb::numpy, T, nb::ndim<2>>(p, {rows, cols}, owner);
}

// Returns (points (N,3) float64, triangles (M,3) int32).
nb::tuple decimate_patch(np_in<double> pts, np_in<int32_t> faces, np_in<double> targets) {
  const size_t nv = pts.shape(0), nf = faces.shape(0);
  const double* pts_data = pts.data();
  const int32_t* faces_data = faces.data();
  const double* targets_data = targets.data();

  size_t onv = 0, onf = 0;
  double* out_pts = nullptr;
  int32_t* out_faces = nullptr;

  // The CGAL pipeline touches no Python state, so drop the GIL for it: the
  // Python caller decimates independent patches on a thread pool, and each
  // call owns its own Sizing tree / Mesh (no shared mutable state).
  {
    nb::gil_scoped_release nogil;

    std::vector<Point_3> orig_pts;
    orig_pts.reserve(nv);
    for (size_t i = 0; i < nv; ++i)
      orig_pts.emplace_back(pts_data[3 * i], pts_data[3 * i + 1], pts_data[3 * i + 2]);
    const Sizing sizing(orig_pts, targets_data);  // built before repair mutates the soup

    std::vector<Point_3> spts = orig_pts;
    std::vector<std::array<std::size_t, 3>> sfaces(nf);
    for (size_t i = 0; i < nf; ++i)
      sfaces[i] = {static_cast<std::size_t>(faces_data[3 * i]),
                   static_cast<std::size_t>(faces_data[3 * i + 1]),
                   static_cast<std::size_t>(faces_data[3 * i + 2])};

    PMP::repair_polygon_soup(spts, sfaces);
    PMP::orient_polygon_soup(spts, sfaces);
    Mesh mesh;
    PMP::polygon_soup_to_polygon_mesh(spts, sfaces, mesh);

    if (mesh.number_of_faces() > 0) {
      Border_map border{&mesh};
      Adaptive_placement adaptive{{}, &sizing};
      using CPlacement = SMS::Constrained_placement<Adaptive_placement, Border_map>;
      // Order by edge length (shortest first); the adaptive placement forbids any
      // collapse once the local target edge length is reached, and only stops
      // globally when no valid collapse remains. Borders (feature curves) are
      // constrained; the normal-change filter blocks flips (preserving curvature).
      SMS::Edge_count_stop_predicate<Mesh> stop(0);
      SMS::Bounded_normal_change_filter<> filter;
      SMS::edge_collapse(mesh, stop,
                         CGAL::parameters::get_cost(SMS::Edge_length_cost<Mesh>())
                             .get_placement(CPlacement(border, adaptive))
                             .filter(filter)
                             .edge_is_constrained_map(border));
      mesh.collect_garbage();
    }

    onv = mesh.number_of_vertices();
    onf = mesh.number_of_faces();
    out_pts = new double[3 * onv];
    for (auto v : mesh.vertices()) {
      const Point_3& p = mesh.point(v);
      const size_t i = static_cast<size_t>(v);
      out_pts[3 * i] = p.x();
      out_pts[3 * i + 1] = p.y();
      out_pts[3 * i + 2] = p.z();
    }
    out_faces = new int32_t[3 * onf];
    size_t fi = 0;
    for (auto f : mesh.faces()) {
      size_t k = 0;
      for (auto v : CGAL::vertices_around_face(mesh.halfedge(f), mesh)) {
        if (k < 3) out_faces[3 * fi + k] = static_cast<int32_t>(static_cast<size_t>(v));
        ++k;
      }
      ++fi;
    }
  }

  return nb::make_tuple(own2d(out_pts, onv, 3), own2d(out_faces, onf, 3));
}

static TopoDS_Shell build_shell(const double* pts, std::size_t np,
                                const int32_t* tris, std::size_t nt,
                                std::size_t& skipped) {
  double vol6 = 0.0;
  for (std::size_t t = 0; t < nt; ++t) {
    const int32_t a = tris[3 * t], b = tris[3 * t + 1], c = tris[3 * t + 2];
    const double* pa = pts + 3 * a;
    const double* pb = pts + 3 * b;
    const double* pc = pts + 3 * c;
    const double cx = pb[1] * pc[2] - pb[2] * pc[1];
    const double cy = pb[2] * pc[0] - pb[0] * pc[2];
    const double cz = pb[0] * pc[1] - pb[1] * pc[0];
    vol6 += pa[0] * cx + pa[1] * cy + pa[2] * cz;
  }
  const bool flip = vol6 < 0.0;

  BRep_Builder builder;
  TopoDS_Shell shell;
  builder.MakeShell(shell);

  std::vector<TopoDS_Vertex> verts(np);
  std::vector<char> have(np, 0);
  auto vertex = [&](int i) -> const TopoDS_Vertex& {
    if (!have[i]) {
      builder.MakeVertex(verts[i],
                         gp_Pnt(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]),
                         Precision::Confusion());
      have[i] = 1;
    }
    return verts[i];
  };

  // Undirected edges are shared between the two faces that meet along them, so
  // cache each once (keyed by the ordered index pair) and hand out the reversed
  // orientation for the opposite half-edge.
  std::unordered_map<uint64_t, TopoDS_Edge> edge_cache;
  auto edge = [&](int i, int j) -> TopoDS_Edge {
    const int lo = std::min(i, j), hi = std::max(i, j);
    const uint64_t key = (static_cast<uint64_t>(lo) << 32) | static_cast<uint32_t>(hi);
    auto it = edge_cache.find(key);
    TopoDS_Edge e;
    if (it == edge_cache.end()) {
      e = BRepBuilderAPI_MakeEdge(vertex(lo), vertex(hi));
      edge_cache.emplace(key, e);
    } else {
      e = it->second;
    }
    return (i < j) ? e : TopoDS::Edge(e.Reversed());
  };

  // Every face is a known planar triangle, so assemble the wire/face directly
  // with BRep_Builder and supply the plane ourselves. This skips the shape
  // analysis BRepBuilderAPI_MakeWire/MakeFace run per triangle (connectivity
  // search + least-squares plane fit), which dominated the export.
  const double tol = Precision::Confusion();
  for (std::size_t t = 0; t < nt; ++t) {
    int a = tris[3 * t], b = tris[3 * t + 1], c = tris[3 * t + 2];
    if (flip) std::swap(b, c);
    if (a == b || b == c || c == a) { ++skipped; continue; }

    const double* pa = pts + 3 * a;
    const double* pb = pts + 3 * b;
    const double* pc = pts + 3 * c;
    const double ux = pb[0] - pa[0], uy = pb[1] - pa[1], uz = pb[2] - pa[2];
    const double vx = pc[0] - pa[0], vy = pc[1] - pa[1], vz = pc[2] - pa[2];
    const double nx = uy * vz - uz * vy;
    const double ny = uz * vx - ux * vz;
    const double nz = ux * vy - uy * vx;
    const double nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (nlen < 1e-12) { ++skipped; continue; }  // degenerate/sliver triangle

    Handle(Geom_Plane) plane =
        new Geom_Plane(gp_Pnt(pa[0], pa[1], pa[2]), gp_Dir(nx, ny, nz));

    TopoDS_Wire wire;
    builder.MakeWire(wire);
    builder.Add(wire, edge(a, b));
    builder.Add(wire, edge(b, c));
    builder.Add(wire, edge(c, a));

    TopoDS_Face face;
    builder.MakeFace(face, plane, tol);
    builder.Add(face, wire);
    builder.Add(shell, face);
  }

  shell.Closed(true);
  return shell;
}

// shells: list of (points (N,3) float64, triangles (M,3) int32); [0] = outer
// peripheral shell, [1:] = nested void shells (cavities).
void write_step(nb::list shells, const std::string& out_path) {
  if (nb::len(shells) == 0)
    throw std::runtime_error("write_step: no shells");

  // The unit is a process-global; set it exactly once so concurrent writers
  // never race on it (see the GIL release below).
  static std::once_flag unit_once;
  std::call_once(unit_once,
                 [] { Interface_Static::SetCVal("write.step.unit", "MM"); });

  // Any OCCT operation below can raise a Standard_Failure. In 8.0 that derives from
  // std::exception, so nanobind would translate it on its own, but we rethrow with the
  // output path and OCCT's own message to make the Python-side error actionable.
  try {
    BRep_Builder sbuilder;
    TopoDS_Solid solid;
    sbuilder.MakeSolid(solid);

    std::size_t skipped = 0;
    for (std::size_t s = 0; s < nb::len(shells); ++s) {
      nb::tuple item = nb::cast<nb::tuple>(shells[s]);
      auto pts = nb::cast<np_in<double>>(item[0]);
      auto tris = nb::cast<np_in<int32_t>>(item[1]);
      TopoDS_Shell shell = build_shell(pts.data(), pts.shape(0),
                                       tris.data(), tris.shape(0), skipped);
      if (s == 0)
        sbuilder.Add(solid, shell);
      else
        sbuilder.Add(solid, TopoDS::Shell(shell.Reversed()));
    }

    // STEP transfer + serialization is the dominant cost and touches no Python
    // state, so drop the GIL: the caller writes independent tissue bodies on a
    // thread pool. Each call owns its own writer/solid; the only shared OCCT
    // global (the unit) is fixed once above before any release.
    bool transfer_ok = false, write_ok = false;
    {
      nb::gil_scoped_release nogil;
      STEPControl_Writer writer;
      transfer_ok = writer.Transfer(solid, STEPControl_AsIs) == IFSelect_RetDone;
      if (transfer_ok)
        write_ok = writer.Write(out_path.c_str()) == IFSelect_RetDone;
    }
    if (!transfer_ok)
      throw std::runtime_error("write_step: STEP transfer failed for " + out_path);
    if (!write_ok)
      throw std::runtime_error("write_step: STEP write failed for " + out_path);
  } catch (const Standard_Failure& e) {
    throw std::runtime_error("write_step: OCCT error for " + out_path + ": " +
                             e.what());
  }
}

NB_MODULE(_mesher_ext, m) {
  m.def("decimate_patch", &decimate_patch, nb::arg("points"), nb::arg("faces"),
        nb::arg("targets"),
        "Decimate one interface patch under a per-vertex target edge-length "
        "field, keeping its feature boundary fixed.");
  m.def("write_step", &write_step, nb::arg("shells"), nb::arg("out_path"),
        "Export one B-rep solid (outer + void shells) to AP214 STEP.");
}
