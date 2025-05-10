# prepare_nfr_model.py

import bpy
import os
import argparse
import sys

# Add bmesh import
import bmesh



class NFRModelPreparer:
    def __init__(self, fbx_path, output_dir, template_obj=None, silence_keywords=None):
        """
        fbx_path       : path to your input FBX
        output_dir     : folder where the prepared OBJ(s) will be written
        template_obj   : (optional) path to a template OBJ for alignment
        silence_keywords: list of substrings; any object whose name contains one
                          of these will be removed (e.g. ['eye','teeth'])
        """
        self.fbx_path = fbx_path
        self.output_dir = output_dir
        self.template_obj = template_obj
        self.silence_keywords = silence_keywords or ['eye', 'eyeball', 'teeth', 'hair']
        os.makedirs(self.output_dir, exist_ok=True)

    def clear_scene(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

    def import_model(self):
        ext = os.path.splitext(self.fbx_path)[1].lower()
        if ext == '.fbx':
            try:
                bpy.ops.import_scene.fbx(filepath=self.fbx_path)
            except Exception as e:
                print(f"FBX import failed ({e}); trying OBJ fallback")
                obj_path = os.path.splitext(self.fbx_path)[0] + '.obj'
                if os.path.exists(obj_path):
                    bpy.ops.import_scene.obj(filepath=obj_path)
                else:
                    raise
        elif ext == '.obj':
            bpy.ops.import_scene.obj(filepath=self.fbx_path)
        else:
            raise ValueError(f"Unsupported model format: {ext}")

    def remove_unwanted(self):
        """Delete any object whose name contains a silence keyword."""
        for obj in bpy.data.objects:
            if any(kw.lower() in obj.name.lower() for kw in self.silence_keywords):
                obj.select_set(True)
            else:
                obj.select_set(False)
        bpy.ops.object.delete()

    def join_meshes(self, target_name='Head'):
        """Join all remaining meshes into one object named `target_name`."""
        meshes = [o for o in bpy.data.objects if o.type == 'MESH']
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        head = bpy.context.active_object
        head.name = target_name
        return head

    def apply_transforms(self, obj):
        """Apply location/rotation/scale so NFR sees a clean mesh."""
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)

    def align_to_template(self, head_obj):
        """Optionally align `head_obj` to the template OBJ’s origin/scale/rotation."""
        if not self.template_obj:
            return
        # import template
        bpy.ops.import_scene.obj(filepath=self.template_obj)
        # assume the template is the only other mesh in scene
        imported = [o for o in bpy.data.objects if o.type=='MESH' and o.name!=head_obj.name][0]
        # copy its transforms
        head_obj.location     = imported.location
        head_obj.rotation_euler = imported.rotation_euler
        head_obj.scale        = imported.scale
        # remove the imported template
        bpy.data.objects.remove(imported, do_unlink=True)

    def export_obj(self, name='your_head.obj'):
        """Export the single mesh to OBJ in output_dir via manual writer."""
        # Find the single mesh object
        head = next(o for o in bpy.data.objects if o.type == 'MESH')
        mesh = head.data

        # Triangulate the mesh using bmesh
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()

        # Build output path
        path = os.path.join(self.output_dir, name)

        # Write OBJ manually
        with open(path, 'w') as f:
            # Write vertices in world space
            for v in mesh.vertices:
                co = head.matrix_world @ v.co
                f.write(f"v {co.x:.6f} {co.y:.6f} {co.z:.6f}\n")
            # Write faces (triangles only)
            for poly in mesh.polygons:
                idxs = [i + 1 for i in poly.vertices]
                if len(idxs) == 3:
                    f.write(f"f {idxs[0]} {idxs[1]} {idxs[2]}\n")

        print(f"Prepared OBJ written to: {path}")
        return path

    def prepare(self):
        """Run the full pipeline and return the exported OBJ path."""
        self.clear_scene()
        self.import_model()
        self.remove_unwanted()
        head = self.join_meshes()
        self.apply_transforms(head)
        self.align_to_template(head)
        out = self.export_obj()
        print(f"Prepared OBJ written to: {out}")
        return out


def main():
    parser = argparse.ArgumentParser(
        description="Prepare an FBX/OBJ for Neural Face Rigging"
    )
    parser.add_argument(
        "--fbx_path", required=True,
        help="Path to the input FBX or OBJ file"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where the prepared OBJ will be written"
    )
    parser.add_argument(
        "--template_obj", default=None,
        help="Optional template OBJ for alignment"
    )
    parser.add_argument(
        "--silence_keywords", nargs="*", default=['eye','eyeball','teeth','hair'],
        help="List of substrings; objects containing any will be removed"
    )
    # Only parse args after Blender’s “--”
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        cli_args = sys.argv[idx+1:]
    else:
        cli_args = sys.argv[1:]
    args = parser.parse_args(cli_args)

    preparer = NFRModelPreparer(
        fbx_path=args.fbx_path,
        output_dir=args.output_dir,
        template_obj=args.template_obj,
        silence_keywords=args.silence_keywords
    )
    preparer.prepare()


if __name__ == "__main__":
    main()