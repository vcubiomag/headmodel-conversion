// MMG3D volume remeshing + CGAL soup repair + OpenCASCADE STEP B-rep export.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_self_intersections.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <limits>
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
#include <BRepCheck_Analyzer.hxx>
#include <STEPControl_Writer.hxx>
#include <STEPControl_StepModelType.hxx>
#include <DESTEP_Parameters.hxx>
#include <StepBasic_Product.hxx>
#include <StepData_StepModel.hxx>
#include <TCollection_HAsciiString.hxx>
#include <UnitsMethods_LengthUnit.hxx>
#include <Standard_Failure.hxx>

#include <mmg/mmg3d/libmmg3d.h>

namespace nb = nanobind;
namespace PMP = CGAL::Polygon_mesh_processing;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;

template <typename T>
using np_in = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

template <typename T>
static nb::ndarray<nb::numpy, T, nb::ndim<2>> own2d(T* p, size_t rows, size_t cols) {
  nb::capsule owner(p, [](void* q) noexcept { delete[] static_cast<T*>(q); });
  return nb::ndarray<nb::numpy, T, nb::ndim<2>>(p, {rows, cols}, owner);
}

// Heal a tissue soup into a closed, non-self-intersecting manifold that
// repair_solids can decompose. It exists for soups whose two walls have met, in
// two distinguishable ways:
//
//   * pinches -- residual non-manifold edges that `orient_polygon_soup` tears into
//     small holes. Filled here with triangulate_hole.
//   * crossings -- clean-manifold walls passing *through* each other. Cleared by
//     remove_self_intersections with preserve_genus(false): a zero-thickness neck
//     is meant to pinch off, which changes genus, so the default (preserve) leaves
//     it; allowing the change is the geometrically correct resolution (the neck was
//     never real volume).
//
// Order matters: orient (which may tear) -> fill the tears closed -> remove the
// crossings (and any the fill introduced). `ok` is whether every self-intersection
// was cleared. Left unchanged with ok=false if the soup will not orient to a mesh.
nb::tuple heal_soup(np_in<double> pts, np_in<int32_t> faces) {
  using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;
  const std::size_t nv = pts.shape(0), nf = faces.shape(0);
  const double* pd = pts.data();
  const int32_t* fd = faces.data();

  std::vector<double> out_p;
  std::vector<int32_t> out_f;
  bool ok = false, manifold = false;
  {
    nb::gil_scoped_release nogil;
    std::vector<Point_3> P;
    P.reserve(nv);
    for (std::size_t i = 0; i < nv; ++i)
      P.emplace_back(pd[3 * i], pd[3 * i + 1], pd[3 * i + 2]);
    std::vector<std::array<std::size_t, 3>> F(nf);
    for (std::size_t i = 0; i < nf; ++i)
      F[i] = {static_cast<std::size_t>(fd[3 * i]),
              static_cast<std::size_t>(fd[3 * i + 1]),
              static_cast<std::size_t>(fd[3 * i + 2])};

    PMP::orient_polygon_soup(P, F);
    if (PMP::is_polygon_soup_a_polygon_mesh(F)) {
      manifold = true;
      Mesh mesh;
      PMP::polygon_soup_to_polygon_mesh(P, F, mesh);

      std::vector<halfedge_descriptor> borders;
      PMP::extract_boundary_cycles(mesh, std::back_inserter(borders));
      for (halfedge_descriptor h : borders)
        if (mesh.is_border(h)) PMP::triangulate_hole(mesh, h);

      ok = PMP::experimental::remove_self_intersections(
          mesh, CGAL::parameters::preserve_genus(false).number_of_iterations(15));
      mesh.collect_garbage();

      out_p.reserve(mesh.number_of_vertices() * 3);
      std::unordered_map<std::size_t, int32_t> remap;
      for (auto v : mesh.vertices()) {
        const Point_3& p = mesh.point(v);
        remap[static_cast<std::size_t>(v)] = static_cast<int32_t>(out_p.size() / 3);
        out_p.push_back(p.x());
        out_p.push_back(p.y());
        out_p.push_back(p.z());
      }
      for (auto f : mesh.faces())
        for (auto vd : CGAL::vertices_around_face(mesh.halfedge(f), mesh))
          out_f.push_back(remap[static_cast<std::size_t>(vd)]);
    }
  }

  if (!manifold) return nb::make_tuple(pts, faces, false);
  const std::size_t onv = out_p.size() / 3, onf = out_f.size() / 3;
  double* pp = new double[out_p.size()];
  std::copy(out_p.begin(), out_p.end(), pp);
  int32_t* tt = new int32_t[out_f.size()];
  std::copy(out_f.begin(), out_f.end(), tt);
  return nb::make_tuple(own2d(pp, onv, 3), own2d(tt, onf, 3), ok);
}

static const char* volume_error_text(PMP::Volume_error_code c) {
  switch (c) {
    case PMP::SURFACE_WITH_SELF_INTERSECTIONS: return "surface self-intersects";
    case PMP::VOLUME_INTERSECTION: return "volumes intersect";
    case PMP::INCOMPATIBLE_ORIENTATION: return "incompatible shell orientation";
    default: return "valid";
  }
}

// One tissue's boundary soup, repaired into solids of [outer shell, *cavities].
//
// The soup arrives already wound so that every normal points out of the tissue
// (see `_boundary_facets`); that is what lets the volume decomposition below tell
// an outer boundary from a cavity, and it is why nothing here needs to guess an
// orientation. Repair may only duplicate points, never move or delete one:
// neighbouring tissues meet at coordinate-identical vertices, so welding or
// shifting a point would silently break conformality with the neighbour's file.
//
// `check_self_intersections` is off by default, and that is a real decision
// rather than a performance one. CGAL's test flags any two triangles that meet,
// which in a head model includes a great deal of legitimate geometry: a tissue
// touching itself along a pinch curve is normal in CHARM (each tissue arrives
// with tens of such edges before anything is remeshed), and two lumps of one
// tissue meeting at an edge are two solids in contact, not an overlap. Enabling
// it as a gate rejects those, so it is an opt-in diagnostic -- useful when
// hunting a suspect tissue, wrong as a precondition.
//
// Note this is also what CGAL's default means: with the tests off it *assumes*
// the mesh has no self-intersections, so SURFACE_WITH_SELF_INTERSECTIONS and
// VOLUME_INTERSECTION become unreachable and the error_codes check below passes
// everything. The real defence against two walls crossing is upstream, in
// remeshing the volume rather than each wall independently; this cannot
// substitute for it.
nb::list repair_solids(np_in<double> pts, np_in<int32_t> faces,
                       bool check_self_intersections) {
  const std::size_t nv = pts.shape(0), nf = faces.shape(0);
  const double* pd = pts.data();
  const int32_t* fd = faces.data();

  struct ShellOut {
    std::vector<double> pts;
    std::vector<int32_t> tris;
  };
  std::vector<std::vector<ShellOut>> solids;
  std::string err;

  {
    nb::gil_scoped_release nogil;

    std::vector<Point_3> spts;
    spts.reserve(nv);
    for (std::size_t i = 0; i < nv; ++i)
      spts.emplace_back(pd[3 * i], pd[3 * i + 1], pd[3 * i + 2]);
    std::vector<std::array<std::size_t, 3>> sfaces(nf);
    for (std::size_t i = 0; i < nf; ++i)
      sfaces[i] = {static_cast<std::size_t>(fd[3 * i]),
                   static_cast<std::size_t>(fd[3 * i + 1]),
                   static_cast<std::size_t>(fd[3 * i + 2])};

    // Splits non-manifold edges and pinch vertices, both by duplicating the point
    // at an identical coordinate. Windings are already coherent, so it has nothing
    // to reverse -- this call is here purely for the split.
    PMP::orient_polygon_soup(spts, sfaces);

    // polygon_soup_to_polygon_mesh's own precondition is compiled out under NDEBUG
    // (CGAL_NO_PRECONDITIONS), so without this gate a soup that is still
    // non-manifold builds a quietly corrupt mesh instead of failing.
    if (!PMP::is_polygon_soup_a_polygon_mesh(sfaces)) {
      err = "soup is still non-manifold after orientation";
    } else {
      Mesh mesh;
      PMP::polygon_soup_to_polygon_mesh(spts, sfaces, mesh);

      if (!CGAL::is_closed(mesh)) {
        std::size_t border = 0;
        for (auto h : mesh.halfedges())
          if (mesh.is_border(h)) ++border;
        err = "boundary is not closed: " + std::to_string(border) +
              " border halfedges over " + std::to_string(mesh.number_of_faces()) +
              " faces (soup had " + std::to_string(nf) + " faces, " +
              std::to_string(nv) + " -> " + std::to_string(spts.size()) +
              " points after splitting)";
      } else {
        auto vol_id =
            mesh.add_property_map<face_descriptor, std::size_t>("f:vol", 0).first;
        auto cc_id =
            mesh.add_property_map<face_descriptor, std::size_t>("f:cc", 0).first;
        std::vector<std::size_t> cc_to_vol;
        std::vector<bool> outward;
        std::vector<PMP::Volume_error_code> codes;

        // Exact predicates decide both the nesting (which shell is a cavity of
        // which lump) and each shell's outward test. That nesting is the whole
        // reason this exports as STEP rather than a triangle soup, so it is not
        // something to settle with a sampled point-in-volume vote.
        //
        const std::size_t nvol = PMP::volume_connected_components(
            mesh, vol_id,
            CGAL::parameters::face_connected_component_map(cc_id)
                .connected_component_id_to_volume_id(std::ref(cc_to_vol))
                .is_cc_outward_oriented(std::ref(outward))
                .do_self_intersection_tests(check_self_intersections)
                .error_codes(std::ref(codes)));

        for (std::size_t v = 0; v < codes.size(); ++v) {
          if (codes[v] != PMP::VALID_VOLUME) {
            err = std::string("volume ") + std::to_string(v) + ": " +
                  volume_error_text(codes[v]);
            break;
          }
        }

        if (err.empty()) {
          const std::size_t ncc = cc_to_vol.size();
          std::vector<std::vector<face_descriptor>> cc_faces(ncc);
          for (auto f : mesh.faces()) cc_faces[cc_id[f]].push_back(f);

          // A shell whose normals point outward bounds its volume from outside;
          // the inward ones are its cavities. Outer first, as the caller expects.
          std::vector<std::vector<std::size_t>> vol_ccs(nvol);
          for (std::size_t c = 0; c < ncc; ++c) {
            auto& group = vol_ccs[cc_to_vol[c]];
            if (outward[c])
              group.insert(group.begin(), c);
            else
              group.push_back(c);
          }

          for (const auto& group : vol_ccs) {
            // A volume with no outward shell is the unbounded exterior, not a body.
            if (group.empty() || !outward[group.front()]) continue;
            std::vector<ShellOut> shells;
            for (std::size_t c : group) {
              ShellOut out;
              std::unordered_map<std::size_t, int32_t> remap;
              for (auto f : cc_faces[c]) {
                int32_t idx[3] = {0, 0, 0};
                std::size_t k = 0;
                for (auto vd : CGAL::vertices_around_face(mesh.halfedge(f), mesh)) {
                  const std::size_t vi = static_cast<std::size_t>(vd);
                  auto it = remap.find(vi);
                  if (it == remap.end()) {
                    const Point_3& p = mesh.point(vd);
                    it = remap.emplace(vi, static_cast<int32_t>(out.pts.size() / 3))
                             .first;
                    out.pts.insert(out.pts.end(), {p.x(), p.y(), p.z()});
                  }
                  if (k < 3) idx[k] = it->second;
                  ++k;
                }
                out.tris.insert(out.tris.end(), {idx[0], idx[1], idx[2]});
              }
              shells.push_back(std::move(out));
            }
            solids.push_back(std::move(shells));
          }
        }
      }
    }
  }

  if (!err.empty()) throw std::runtime_error("repair_solids: " + err);

  nb::list out;
  for (auto& shells : solids) {
    nb::list group;
    for (auto& s : shells) {
      const size_t n = s.pts.size() / 3, m = s.tris.size() / 3;
      double* p = new double[s.pts.size()];
      std::copy(s.pts.begin(), s.pts.end(), p);
      int32_t* t = new int32_t[s.tris.size()];
      std::copy(s.tris.begin(), s.tris.end(), t);
      group.append(nb::make_tuple(own2d(p, n, 3), own2d(t, m, 3)));
    }
    out.append(group);
  }
  return out;
}

static TopoDS_Shell build_shell(const double* pts, std::size_t np,
                                const int32_t* tris, std::size_t nt,
                                const std::string& what) {
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
  // orientation for the opposite half-edge. The cache will just as happily serve
  // a third or fourth face, so tally the traversals and check them below rather
  // than trust the caller to have handed us a manifold.
  std::unordered_map<uint64_t, TopoDS_Edge> edge_cache;
  std::unordered_map<uint64_t, std::pair<int, int>> edge_use;  // key -> (count, balance)
  auto edge = [&](int i, int j) -> TopoDS_Edge {
    const int lo = std::min(i, j), hi = std::max(i, j);
    const uint64_t key = (static_cast<uint64_t>(lo) << 32) | static_cast<uint32_t>(hi);
    auto& use = edge_use[key];
    use.first += 1;
    use.second += (i < j) ? 1 : -1;
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
  std::size_t repeated = 0, slivers = 0;
  for (std::size_t t = 0; t < nt; ++t) {
    int a = tris[3 * t], b = tris[3 * t + 1], c = tris[3 * t + 2];
    if (flip) std::swap(b, c);
    if (a == b || b == c || c == a) { ++repeated; continue; }

    const double* pa = pts + 3 * a;
    const double* pb = pts + 3 * b;
    const double* pc = pts + 3 * c;
    const double ux = pb[0] - pa[0], uy = pb[1] - pa[1], uz = pb[2] - pa[2];
    const double vx = pc[0] - pa[0], vy = pc[1] - pa[1], vz = pc[2] - pa[2];
    const double nx = uy * vz - uz * vy;
    const double ny = uz * vx - ux * vz;
    const double nz = ux * vy - uy * vx;
    const double nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (nlen < 1e-12) { ++slivers; continue; }  // degenerate/sliver triangle

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

  // Dropping a triangle would leave a hole in something we are about to declare
  // closed, so a shell that needs dropping is a caller bug, not a thing to patch
  // over quietly.
  if (repeated || slivers)
    throw std::runtime_error(
        "write_step: " + what + " has " + std::to_string(repeated) +
        " repeated-vertex and " + std::to_string(slivers) +
        " zero-area triangles; dropping them would leave the shell open");

  // Closed(true) is a claim OCCT takes at face value and STEP then records as a
  // CLOSED_SHELL, so establish it rather than assert it: a closed, coherently
  // oriented manifold uses every edge exactly twice, once in each direction.
  std::size_t bad_count = 0, bad_dir = 0;
  for (const auto& [key, use] : edge_use) {
    if (use.first != 2)
      ++bad_count;
    else if (use.second != 0)
      ++bad_dir;
  }
  if (bad_count || bad_dir)
    throw std::runtime_error(
        "write_step: " + what + " is not a closed oriented manifold (" +
        std::to_string(bad_count) + " edges not used exactly twice, " +
        std::to_string(bad_dir) + " traversed twice the same way)");

  shell.Closed(true);
  return shell;
}

// Borrowed view of one shell's arrays. The nb::list argument keeps the numpy
// objects alive for the whole call, so these stay valid after the GIL is
// released (unlike the nb::ndarray handles, whose refcounting needs the GIL).
struct ShellView {
  const double* pts;
  std::size_t np;
  const int32_t* tris;
  std::size_t nt;
};

struct SolidView {
  std::string name;
  std::vector<ShellView> shells;  // [0] = outer, [1:] = cavities
};

// solids: list of (name, shells), where shells is a list of
// (points (N,3) float64, triangles (M,3) int32); shells[0] is the outer
// peripheral shell and shells[1:] are its nested void shells (cavities).
// Every solid becomes its own STEP product named `name` in the one output file.
void write_step(nb::list solids, const std::string& out_path) {
  if (nb::len(solids) == 0)
    throw std::runtime_error("write_step: no solids");

  std::vector<SolidView> views;
  views.reserve(nb::len(solids));
  for (std::size_t i = 0; i < nb::len(solids); ++i) {
    nb::tuple entry = nb::cast<nb::tuple>(solids[i]);
    SolidView view;
    view.name = nb::cast<std::string>(entry[0]);
    nb::list shells = nb::cast<nb::list>(entry[1]);
    if (nb::len(shells) == 0)
      throw std::runtime_error("write_step: no shells for solid " + view.name);
    view.shells.reserve(nb::len(shells));
    for (std::size_t s = 0; s < nb::len(shells); ++s) {
      nb::tuple item = nb::cast<nb::tuple>(shells[s]);
      auto pts = nb::cast<np_in<double>>(item[0]);
      auto tris = nb::cast<np_in<int32_t>>(item[1]);
      view.shells.push_back(
          {pts.data(), pts.shape(0), tris.data(), tris.shape(0)});
    }
    views.push_back(std::move(view));
  }

  // Any OCCT operation below can raise a Standard_Failure. In 8.0 that derives from
  // std::exception, so nanobind would translate it on its own, but we rethrow with the
  // output path and OCCT's own message to make the Python-side error actionable.
  try {
    bool transfer_ok = true, write_ok = false, brep_invalid = false;
    std::string failed;
    // None of the OCCT work touches Python state, so drop the GIL for all of it:
    // the caller writes independent tissue files from a process pool (OCCT's
    // transfer serializes on in-process globals, so threads would not help), and
    // each call owns its own writer, params and shapes.
    {
      nb::gil_scoped_release nogil;

      // Constructed first: its ctor runs the idempotent STEPControl_Controller::Init()
      // that registers the statics InitFromStatic reads back. The 5-arg Transfer
      // overload below then takes these params explicitly; the 3-arg one would
      // re-init the model from the process-global statics on every call.
      STEPControl_Writer writer;
      DESTEP_Parameters params;
      params.InitFromStatic();
      params.WriteUnit = UnitsMethods_LengthUnit_Millimeter;
      // Every face here is a plane, so a pcurve is just the exact linear preimage of the
      // 3D line and any reader recomputes it for free. Writing them doubles the entity
      // count (and the file) for nothing: 591 MB -> 262 MB across a subject's tissues.
      params.WriteSurfaceCurMode = false;

      Handle(StepData_StepModel) model = writer.Model();
      for (const SolidView& view : views) {
        BRep_Builder sbuilder;
        TopoDS_Solid solid;
        sbuilder.MakeSolid(solid);
        for (std::size_t s = 0; s < view.shells.size(); ++s) {
          const ShellView& sv = view.shells[s];
          const std::string what =
              view.name + (s == 0 ? " outer shell" : " cavity " + std::to_string(s));
          TopoDS_Shell shell = build_shell(sv.pts, sv.np, sv.tris, sv.nt, what);
          if (s == 0)
            sbuilder.Add(solid, shell);
          else
            sbuilder.Add(solid, TopoDS::Shell(shell.Reversed()));
        }
        solid.Closed(true);

        // build_shell only ever proved the combinatorics -- every edge used twice,
        // once each way. That says nothing about where the faces actually are, so
        // ask OCCT to look at the geometry before this becomes a file. This is the
        // last gate before Ansys, and OCCT is the same kernel family Ansys will
        // read the result with.
        if (!BRepCheck_Analyzer(solid).IsValid()) {
          brep_invalid = true;
          failed = view.name;
          break;
        }

        // Transferring onto the same writer appends another root product, so all
        // of this tissue's solids end up in the one file.
        const int before = model->NbEntities();
        if (writer.Transfer(solid, STEPControl_AsIs, params) != IFSelect_RetDone) {
          transfer_ok = false;
          failed = view.name;
          break;
        }

        // Name the product this transfer appended. Not via params.WriteProductName:
        // that route appends the actor's assembly index to the name, and the actor
        // is a process-global we cannot reset, so the suffix would depend on what
        // else this worker had already written. MakeSDR creates the one product
        // (using the same string for its id and name) before transferring any
        // geometry, so it is at the front of the range -- stop on it rather than
        // walk the millions of geometry entities behind it.
        Handle(TCollection_HAsciiString) pname =
            new TCollection_HAsciiString(view.name.c_str());
        for (int e = before + 1, n = model->NbEntities(); e <= n; ++e) {
          Handle(StepBasic_Product) product =
              Handle(StepBasic_Product)::DownCast(model->Value(e));
          if (!product.IsNull()) {
            product->SetId(pname);
            product->SetName(pname);
            break;
          }
        }
      }
      if (transfer_ok && !brep_invalid)
        write_ok = writer.Write(out_path.c_str()) == IFSelect_RetDone;
    }
    if (brep_invalid)
      throw std::runtime_error("write_step: B-rep for solid " + failed + " in " +
                               out_path + " is geometrically invalid (BRepCheck)");
    if (!transfer_ok)
      throw std::runtime_error("write_step: STEP transfer failed for solid " +
                               failed + " in " + out_path);
    if (!write_ok)
      throw std::runtime_error("write_step: STEP write failed for " + out_path);
  } catch (const Standard_Failure& e) {
    throw std::runtime_error("write_step: OCCT error for " + out_path + ": " +
                             e.what());
  }
}

// Coarsen a multi-material tetrahedral mesh with MMG3D, preserving every material
// interface. Working on the volume rather than on each interface surface is what
// makes the result conformal and non-self-intersecting for the folded tissues: a
// tissue's two walls are remeshed together and cannot cross. `hausd`
// bounds how far a boundary may move (the fidelity/size knob); `hmax` caps
// edge length, `hmin` floors it, `hgrad` bounds size gradation; non-positive
// hmax/hmin/hgrad keep MMG's default. Returns (points, tets 0-based, tet labels).
nb::tuple mmg_remesh(np_in<double> pts, np_in<int32_t> tets, np_in<int32_t> labels,
                     double hausd, double hmax, double hmin, double hgrad,
                     int verbose, int aniso, int angle_detect, int nofem) {
  const std::size_t np = pts.shape(0), ne = tets.shape(0);
  const double* pd = pts.data();
  const int32_t* td = tets.data();
  const int32_t* ld = labels.data();

  std::vector<double> out_p;
  std::vector<int32_t> out_t, out_l;
  std::string err;
  bool low_failure = false;
  {
    nb::gil_scoped_release nogil;
    MMG5_pMesh mesh = nullptr;
    MMG5_pSol met = nullptr;
    MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh, MMG5_ARG_ppMet, &met,
                    MMG5_ARG_end);

    bool ok = MMG3D_Set_meshSize(mesh, static_cast<MMG5_int>(np),
                                 static_cast<MMG5_int>(ne), 0, 0, 0, 0) == 1;
    if (ok) {
      std::vector<double> vc(pd, pd + 3 * np);  // Set_vertices takes non-const
      std::vector<MMG5_int> vr(np, 0);
      ok = MMG3D_Set_vertices(mesh, vc.data(), vr.data()) == 1;
      std::vector<MMG5_int> tt(4 * ne), tr(ne);
      for (std::size_t i = 0; i < 4 * ne; ++i)
        tt[i] = static_cast<MMG5_int>(td[i]) + 1;  // 0-based -> MMG's 1-based
      for (std::size_t i = 0; i < ne; ++i) tr[i] = static_cast<MMG5_int>(ld[i]);
      if (ok) ok = MMG3D_Set_tetrahedra(mesh, tt.data(), tr.data()) == 1;

      MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_verbose, verbose);
      // anisosize: build an anisotropic size map from curvature so a sulcus can be
      // spanned by long thin triangles along its low-curvature axis -- far fewer
      // faces than isotropic for the same chord error. angle: ridge detection;
      // biological interfaces are smooth, so off lets those coarsen. nofem: skip the
      // FE-quality passes -- Ansys remeshes anyway, we only need valid boundaries.
      MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_anisosize, aniso);
      MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_angle, angle_detect);
      MMG3D_Set_iparameter(mesh, met, MMG3D_IPARAM_nofem, nofem);
      MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hausd, hausd);
      if (hmax > 0) MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hmax, hmax);
      if (hmin > 0) MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hmin, hmin);
      if (hgrad > 0) MMG3D_Set_dparameter(mesh, met, MMG3D_DPARAM_hgrad, hgrad);

      if (ok) {
        // A strong failure leaves no usable mesh. A low failure returns one that is
        // valid but did not meet every requested criterion; the whole pipeline now
        // rests on this remesh, so it is reported rather than passed off as success.
        const int ier = MMG3D_mmg3dlib(mesh, met);
        if (ier == MMG5_STRONGFAILURE) {
          err = "MMG3D_mmg3dlib failed (could not remesh)";
        } else if (ier == MMG5_LOWFAILURE) {
          low_failure = true;
        }
      } else {
        err = "MMG3D input rejected (Set_* failed)";
      }
    } else {
      err = "MMG3D_Set_meshSize failed";
    }

    if (err.empty()) {
      MMG5_int onp = 0, one = 0;
      MMG3D_Get_meshSize(mesh, &onp, &one, nullptr, nullptr, nullptr, nullptr);
      out_p.resize(3 * static_cast<std::size_t>(onp));
      MMG3D_Get_vertices(mesh, out_p.data(), nullptr, nullptr, nullptr);
      std::vector<MMG5_int> ott(4 * static_cast<std::size_t>(one)), otr(one);
      MMG3D_Get_tetrahedra(mesh, ott.data(), otr.data(), nullptr);
      out_t.resize(4 * static_cast<std::size_t>(one));
      out_l.resize(one);
      for (std::size_t i = 0; i < out_t.size(); ++i)
        out_t[i] = static_cast<int32_t>(ott[i]) - 1;  // MMG 1-based -> 0-based
      for (std::size_t i = 0; i < out_l.size(); ++i)
        out_l[i] = static_cast<int32_t>(otr[i]);
    }

    MMG3D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh, MMG5_ARG_ppMet, &met,
                   MMG5_ARG_end);
  }

  if (!err.empty()) throw std::runtime_error("mmg_remesh: " + err);
  if (low_failure)
    PyErr_WarnEx(PyExc_RuntimeWarning,
                 "MMG3D returned a low failure: the mesh is usable but did not meet "
                 "every requested criterion. Check the boundary quality.",
                 1);

  const std::size_t onp = out_p.size() / 3, one = out_t.size() / 4;
  double* pp = new double[out_p.size()];
  std::copy(out_p.begin(), out_p.end(), pp);
  int32_t* tt = new int32_t[out_t.size()];
  std::copy(out_t.begin(), out_t.end(), tt);
  int32_t* ll = new int32_t[out_l.size()];
  std::copy(out_l.begin(), out_l.end(), ll);
  nb::capsule lo(ll, [](void* q) noexcept { delete[] static_cast<int32_t*>(q); });
  return nb::make_tuple(own2d(pp, onp, 3), own2d(tt, one, 4),
                        nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(ll, {one}, lo));
}

NB_MODULE(_mesher_ext, m) {
  m.def("mmg_remesh", &mmg_remesh, nb::arg("points"), nb::arg("tets"),
        nb::arg("labels"), nb::arg("hausd"), nb::arg("hmax") = 0.0,
        nb::arg("hmin") = 0.0, nb::arg("hgrad") = 0.0, nb::arg("verbose") = 1,
        nb::arg("aniso") = 0, nb::arg("angle_detect") = 1, nb::arg("nofem") = 0,
        "Coarsen a multi-material tet mesh with MMG3D under a Hausdorff bound, "
        "preserving material interfaces. `aniso` builds an anisotropic size map "
        "(fewer faces at equal fidelity); `angle_detect` toggles ridge detection; "
        "`nofem` skips FE-quality passes. Returns (points, tets, tet labels).");
  m.def("heal_soup", &heal_soup, nb::arg("points"), nb::arg("faces"),
        "Heal a tissue soup into a closed non-self-intersecting manifold: "
        "orient, fill pinch tears, remove wall crossings. Returns (points, tris, "
        "ok); ok is whether all self-intersections cleared. Unchanged with ok=False "
        "if the soup will not orient to a mesh.");
  m.def("repair_solids", &repair_solids, nb::arg("points"), nb::arg("faces"),
        nb::arg("check_self_intersections") = false,
        "Split one tissue's outward-wound boundary soup into valid solids, each "
        "returned as [(outer points, tris), *(cavity points, tris)].");
  m.def("write_step", &write_step, nb::arg("solids"), nb::arg("out_path"),
        "Export named B-rep solids (each outer + void shells) to one AP214 "
        "STEP file, one named product per solid.");
}
