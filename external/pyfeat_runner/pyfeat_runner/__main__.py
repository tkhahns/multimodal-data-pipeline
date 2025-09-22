import argparse
import json
import sys
import numpy as np

def run(video_path: str) -> dict:
    try:
        from feat import Detector
    except Exception as e:
        return {"error": f"ImportError: {e}"}

    try:
        det = Detector()
        result = det.detect_video(video_path)
        df = result.to_pandas() if hasattr(result, 'to_pandas') else result

        features = {}

        au_map = {
            'pf_au01': 'AU01_r','pf_au02': 'AU02_r','pf_au04': 'AU04_r','pf_au05': 'AU05_r','pf_au06': 'AU06_r',
            'pf_au07': 'AU07_r','pf_au09': 'AU09_r','pf_au10': 'AU10_r','pf_au11': 'AU11_r','pf_au12': 'AU12_r',
            'pf_au14': 'AU14_r','pf_au15': 'AU15_r','pf_au17': 'AU17_r','pf_au20': 'AU20_r','pf_au23': 'AU23_r',
            'pf_au24': 'AU24_r','pf_au25': 'AU25_r','pf_au26': 'AU26_r','pf_au28': 'AU28_r','pf_au43': 'AU43_r'
        }
        for out_key, col in au_map.items():
            if col in df.columns:
                val = df[col].values
                features[out_key] = float(np.nanmean(val)) if val.size else float('nan')

        emo_cols = ['anger','disgust','fear','happiness','sadness','surprise','neutral']
        for emo in emo_cols:
            if emo in df.columns:
                val = df[emo].values
                features[f'pf_{emo}'] = float(np.nanmean(val)) if val.size else float('nan')

        for col_name, out_prefix in [
            ('face_x', 'pf_facerectx'), ('face_y', 'pf_facerecty'),
            ('face_w', 'pf_facerectwidth'), ('face_h', 'pf_facerectheight'), ('face_score', 'pf_facescore')
        ]:
            if col_name in df.columns:
                val = df[col_name].values
                features[out_prefix] = float(np.nanmean(val)) if val.size else float('nan')

        for ang in ['pitch','roll','yaw']:
            if ang in df.columns:
                val = df[ang].values
                features[f'pf_{ang}'] = float(np.nanmean(val)) if val.size else float('nan')

        for axis in ['x','y','z']:
            coln = f'landmark_{axis}'
            if coln in df.columns:
                val = df[coln].values
                features[f'pf_{axis}'] = float(np.nanmean(val)) if val.size else float('nan')

        return {"features": features}
    except Exception as e:
        return {"error": str(e)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Py-Feat on a video and print JSON features")
    parser.add_argument("video", help="Path to video file")
    args = parser.parse_args(argv)

    out = run(args.video)
    json.dump(out, sys.stdout)

if __name__ == "__main__":
    main()
