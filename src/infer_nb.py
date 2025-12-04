import argparse
import joblib
import pandas as pd
from lol_pipeline import feature_engineering


def parse_args():
    p = argparse.ArgumentParser(description="Inferência com GaussianNB")
    p.add_argument("--model_path", default="outputs/GaussianNB_pipeline.joblib")
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_path", default="predictions_nb.csv")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Carregando modelo: {args.model_path}")
    pipe = joblib.load(args.model_path)

    print(f"[INFO] Carregando dados: {args.data_path}")
    df = pd.read_csv(args.data_path)

    print("[INFO] Aplicando feature engineering...")
    df_fe = feature_engineering(df)

    X = df_fe.drop(columns=["blueWins", "gameId"], errors="ignore")

    print("[INFO] Gerando predições...")
    df_fe["pred"] = pipe.predict(X)
    df_fe["proba"] = pipe.predict_proba(X)[:, 1]

    df_fe.to_csv(args.output_path, index=False)
    print(f"[INFO] Arquivo gerado: {args.output_path}")


if __name__ == "__main__":
    main()
