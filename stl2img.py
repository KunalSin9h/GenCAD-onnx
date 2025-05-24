import os
import argparse
from OCC.Core.StlAPI import StlAPI_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.AIS import AIS_Shape
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Display.SimpleGui import init_display


def render_stl_files(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)

    display, *_ = init_display()
    view = display.View

    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(".stl"):
            continue

        fpath = os.path.join(src_dir, fname)
        out_png = os.path.join(dst_dir, f"{os.path.splitext(fname)[0]}.png")

        shape = TopoDS_Shape()
        if not StlAPI_Reader().Read(shape, fpath):
            print(f"Failed: {fpath}")
            continue

        BRepMesh_IncrementalMesh(shape, 0.1).Perform()

        ais = AIS_Shape(shape)
        display.Context.Display(ais, True)
        display.FitAll()
        view.Dump(out_png)
        display.Context.Erase(ais)

        print(f"Saved: {out_png}")

    display.Finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render PNG previews of all STL files in a folder.")
    p.add_argument("src", help="Path to folder containing .stl files")
    p.add_argument("dst", help="Output folder for .png renders")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_stl_files(args.src, args.dst)
