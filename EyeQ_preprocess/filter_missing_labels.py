import argparse
import os
import pandas as pd


def image_to_png_name(image_name):
    base, _ = os.path.splitext(image_name)
    return base + '.png'


def filter_csv(csv_path, images_dir, out_path):
    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        raise ValueError("CSV missing required 'image' column")

    kept_rows = []
    missing = []

    for _, row in df.iterrows():
        image_name = str(row["image"])
        png_name = image_to_png_name(image_name)
        image_path = os.path.join(images_dir, png_name)
        if os.path.exists(image_path):
            kept_rows.append(row)
        else:
            missing.append(image_path)

    out_df = pd.DataFrame(kept_rows)
    out_df.to_csv(out_path, index=False)

    print("Input rows : {}".format(len(df)))
    print("Kept rows  : {}".format(len(out_df)))
    print("Missing    : {}".format(len(missing)))
    if missing:
        print("First missing: {}".format(missing[0]))


def main():
    parser = argparse.ArgumentParser(description="Filter CSV rows with missing preprocessed images")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--images-dir", required=True, help="Directory with preprocessed PNG images")
    parser.add_argument("--out", required=True, help="Output CSV file")
    args = parser.parse_args()

    filter_csv(args.csv, args.images_dir, args.out)


if __name__ == "__main__":
    main()
