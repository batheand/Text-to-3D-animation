

#!/usr/bin/env python3
"""
Command-line tool for running Neural Face Rigging (NFR) on prepared head meshes,
and (optionally) extracting expression parameter vectors (blend weights) per pose.

Based on:
- NFR transfer API: NFR can retarget facial animations to arbitrary meshes citeturn1search1
- Expression space: human‑interpretable parameters for artistic controls citeturn0search6
"""
import argparse
import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import diffusion_net

class NFRRiggingPipeline:
    """
    Wrapper around Neural Face Rigging to batch‑rig meshes and extract expression codes.
    """
    def __init__(
        self,
        nfr_repo_path: str,
        neutral_obj: str,
        target_obj: str,
        device: str = "cpu",
        project: bool = False,
    ):
        # Make sure NFR code is importable
        sys.path.append(nfr_repo_path)
        from myutils import Mesh
        from deformation_transfer import Transfer

        # Load meshes
        self.neutral = Mesh(neutral_obj)
        self.target  = Mesh(target_obj)

        # Build the Transfer object
        self.transfer = Transfer(
            source=self.neutral,
            target=self.target,
            project=project,
            device=device,
        )

    def rig_all(
        self,
        poses_dir: str,
        output_dir: str,
        extension: str = ".obj",
    ) -> dict:
        """
        Rig every mesh in poses_dir matching extension, saving into output_dir.
        Returns a dict: input_filename -> output_path.
        """
        from myutils import Mesh  # ensure import after sys.path update
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        for fname in sorted(os.listdir(poses_dir)):
            if not fname.lower().endswith(extension):
                continue
            src = os.path.join(poses_dir, fname)
            dst = os.path.join(output_dir, fname)
            pose = Mesh(src)
            rigged = self.transfer(pose)
            rigged.save(dst)
            results[fname] = dst
        return results

    def extract_weights(self, pose_obj: str):
        """
        Extract expression parameters (blend weights) for a single pose.
        TODO: Modify this method to call the NFR encoder or extend Transfer to return codes.
        Example (pseudo-code):
            expr_code = self.transfer.encoder(pose)
        """
        from myutils import Mesh
        pose = Mesh(pose_obj)
        # TODO: Replace the following with actual encoder call:
        raise NotImplementedError("extract_weights() must be implemented based on NFR internals")

def main():
    parser = argparse.ArgumentParser(description="Run NFR rigging or weight extraction")
    parser.add_argument("--nfr_repo",    required=True, help="Path to NFR_pytorch repo")
    parser.add_argument("--neutral",     required=True, help="Neutral OBJ from NFR data")
    parser.add_argument("--target",      required=True, help="Prepared head OBJ")
    parser.add_argument("--poses_dir",   required=True, help="Directory of pose OBJs")
    parser.add_argument("--output_dir",  required=True, help="Where to save rigged OBJs")
    parser.add_argument("--device",      default="cpu", help="cpu or cuda:0")
    parser.add_argument("--project",     action="store_true", help="Enable tangent-space projection")
    parser.add_argument("--mode",        choices=["rig_all","extract_weights"], default="rig_all",
                        help="Whether to rig all poses or extract weights for each")
    parser.add_argument("--weights_out", default=None,
                        help="(For extract_weights) JSON path to dump expression codes")
    args = parser.parse_args()

    pipeline = NFRRiggingPipeline(
        nfr_repo_path=args.nfr_repo,
        neutral_obj=args.neutral,
        target_obj=args.target,
        device=args.device,
        project=args.project,
    )

    if args.mode == "rig_all":
        results = pipeline.rig_all(args.poses_dir, args.output_dir)
        print(json.dumps(results, indent=2))
    else:
        if not args.weights_out:
            parser.error("--weights_out is required for extract_weights mode")
        os.makedirs(os.path.dirname(args.weights_out), exist_ok=True)
        weights_map = {}
        for fname in sorted(os.listdir(args.poses_dir)):
            if not fname.lower().endswith(".obj"):
                continue
            src = os.path.join(args.poses_dir, fname)
            weights = pipeline.extract_weights(src)
            weights_map[fname] = weights
        with open(args.weights_out, "w") as f:
            json.dump(weights_map, f, indent=2)
        print(f"Extracted weights for {len(weights_map)} poses to {args.weights_out}")

if __name__ == "__main__":
    main()