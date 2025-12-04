from __future__ import annotations
import argparse, logging, os, sys
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report, confusion_matrix

# Tenta importar SMOTE (não necessário para este dataset porque ele é balanceado)
try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# Logging - detalhado
logger = logging.getLogger("lol_pipeline")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline final - Previsão do vencedor (LoL)")
    p.add_argument("--data_path", required=True, help="Caminho do CSV (dataset fornecido)")
    p.add_argument("--target", default="blueWins", help="Nome da coluna alvo (padrão: blueWins)")
    p.add_argument("--output_dir", default="outputs", help="Onde salvar os artefatos")
    p.add_argument("--test_size", type=float, default=0.2, help="Proporção da base de teste")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--run_tuning", action="store_true", help="Executar tuning com GridSearch para DecisionTree")
    p.add_argument("--n_jobs", type=int, default=-1, help="Trabalhos paralelos para CV/GridSearch")
    return p.parse_args()


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Carregando dados de {path}")
    df = pd.read_csv(path)
    logger.info(f"DataFrame carregado com shape: {df.shape}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando feature engineering determinística (sem placeholders).")
    df = df.copy()
    # cria diferenças entre blue e red para cada estatística
    pairs = [
        ("Kills", "Kills"),
        ("Deaths", "Deaths"),
        ("Assists", "Assists"),
        ("TowersDestroyed", "TowersDestroyed"),
        ("Dragons", "Dragons"),
        ("Heralds", "Heralds"),
        ("EliteMonsters", "EliteMonsters"),
        ("WardsPlaced", "WardsPlaced"),
        ("WardsDestroyed", "WardsDestroyed"),
        ("TotalGold", "TotalGold"),
    ]
    created = []
    for b_suffix, r_suffix in pairs:
        blue_col = f"blue{b_suffix}"
        red_col = f"red{r_suffix}"
        if blue_col in df.columns and red_col in df.columns:
            new = f"{b_suffix.lower()}_diff"
            df[new] = df[blue_col] - df[red_col]
            created.append(new)
            logger.debug(f"Feature criada {new} a partir de {blue_col} - {red_col}")

    # first blood: se blueFirstBlood existe, use como feature
    if "blueFirstBlood" in df.columns:
        df["first_blood_flag"] = df["blueFirstBlood"].astype(int)
        created.append("first_blood_flag")
        logger.debug("Criado first_blood_flag a partir de blueFirstBlood")

    drop_cols = [c for c in df.columns if (c.startswith("blue") or c.startswith("red")) and
                 (c not in ["blueFirstBlood", "blueWins"])]
    drop_cols = [c for c in drop_cols if c != "blueWins"]
    logger.info(f"Removendo {len(drop_cols)} colunas brutas por time para manter o dataset compacto para modelagem.")
    df_model = df.drop(columns=drop_cols)

    # Ordenamento/logging final
    logger.info(f"Feature engineering criou {len(created)} features: {created}")
    return df_model


def prepare_data(df: pd.DataFrame, target: str, test_size: float, random_state: int
                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger.info("Preparando os dados: separando X/y e treino/teste.")
    if target not in df.columns:
        raise ValueError(f"Alvo '{target}' não encontrado nas colunas do dataframe: {df.columns.tolist()}")
    y = df[target].astype(int)
    X = df.drop(columns=[target, "gameId"] if "gameId" in df.columns else [target])
    # Mantém apenas colunas numéricas (após engenharia determinística, features são numéricas)
    X = X.select_dtypes(include=["int64", "float64"])
    logger.info(f"Shape das features do modelo: {X.shape}; distribuição do alvo:\n{y.value_counts().to_dict()}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        stratify=y, random_state=random_state)
    logger.info(f"Shape de treino: {X_train.shape}, shape de teste: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def make_preprocessor(numeric_features: List[str]) -> ColumnTransformer:
    logger.info("Criando pipeline de pré-processamento numérico (imputação por mediana + standard scaler).")
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preproc = ColumnTransformer(transformers=[("num", num_transformer, numeric_features)], remainder="drop")
    return preproc


def train_and_evaluate(X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.Series, y_test: pd.Series,
                       preproc: ColumnTransformer,
                       output_dir: str, random_state: int, n_jobs: int) -> Dict[str, dict]:
    os.makedirs(output_dir, exist_ok=True)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=random_state)
    }
    pipelines = {name: Pipeline([("preproc", preproc), ("clf", clf)]) for name, clf in models.items()}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}
    # Cross-val AUC para cada modelo
    for name, pipe in pipelines.items():
        logger.info(f"Cross-validando (AUC) {name} ...")
        try:
            scores = cross_val_score(pipe, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=n_jobs)
            results[name] = {"cv_auc_mean": float(scores.mean()), "cv_auc_std": float(scores.std())}
            logger.info(f"{name} CV AUC: média={scores.mean():.4f} desvio={scores.std():.4f}")
        except Exception as e:
            logger.exception(f"Cross-val falhou para {name}: {e}")
            results[name] = {"cv_auc_mean": None, "cv_auc_std": None}

    # Adequa para todo o conjunto de treino e avalia no teste
    plt.figure(figsize=(8,6))
    for name, pipe in pipelines.items():
        logger.info(f"Treinando pipeline final para {name} ...")
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, os.path.join(output_dir, f"{name}_pipeline.joblib"))
        logger.info(f"Pipeline salva: {os.path.join(output_dir, f'{name}_pipeline.joblib')}")

        # predict proba
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            try:
                y_proba = pipe.decision_function(X_test)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)
            except Exception:
                y_proba = pipe.predict(X_test)

        y_pred = pipe.predict(X_test)
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        results[name].update({"test_auc": float(auc), "test_acc": float(acc)})
        results[name].update({"classification_report": classification_report(y_test, y_pred, output_dict=True)})

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["neg","pos"], columns=["pred_neg","pred_pos"])
        cm_df.to_csv(os.path.join(output_dir, f"{name}_confusion_matrix.csv"))
        logger.info(f"Matriz de confusão salva para {name}")

    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC - Conjunto de teste"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png")); plt.close()
    logger.info("Plot das curvas ROC salvo.")

    # Salva resumo dos resultados
    summary_rows = []
    for name, info in results.items():
        summary_rows.append({
            "model": name,
            "cv_auc_mean": info.get("cv_auc_mean"),
            "cv_auc_std": info.get("cv_auc_std"),
            "test_auc": info.get("test_auc"),
            "test_acc": info.get("test_acc")
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "models_summary.csv"), index=False)
    logger.info("Arquivo models_summary.csv salvo.")
    return results


def tune_decision_tree(X_train, y_train, preproc, output_dir, cv_splits=5, n_jobs=-1, random_state=42):
    logger.info("Ajustando DecisionTree via GridSearchCV ...")
    pipe = Pipeline([("preproc", preproc), ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=random_state))])
    param_grid = {"clf__max_depth": [3,5,8,12,None], "clf__min_samples_leaf": [1,2,5,10]}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    gs = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=n_jobs, verbose=1)
    gs.fit(X_train, y_train)
    joblib.dump(gs.best_estimator_, os.path.join(output_dir, "DecisionTree_best_pipeline.joblib"))
    logger.info(f"GridSearch concluído. Melhores parâmetros: {gs.best_params_}. Pipeline salva.")
    return gs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_data(args.data_path)

    # Confirma uso de blueWins como alvo
    if args.target not in df.columns:
        logger.warning(f"O alvo solicitado {args.target} não foi encontrado no dataset. Usando 'blueWins' se existir.")
        if "blueWins" in df.columns:
            args.target = "blueWins"
        else:
            raise ValueError("Nenhuma coluna alvo válida encontrada. Especifique um --target que exista no CSV.")

    # Engenharia de features
    df_fe = feature_engineering(df)

    # Artefatos de análise exploratória
    pd.DataFrame(df_fe.describe(include='all').T).to_csv(os.path.join(args.output_dir, "data_describe.csv"))
    df_fe.isna().sum().sort_values(ascending=False).to_csv(os.path.join(args.output_dir, "missing_values.csv"))
    df_fe[args.target].value_counts().to_csv(os.path.join(args.output_dir, "target_distribution.csv"))

    # Prepara dados
    X_train, X_test, y_train, y_test = prepare_data(df_fe, args.target, args.test_size, args.random_state)
    numeric_features = X_train.columns.tolist()
    preproc = make_preprocessor(numeric_features)

    # Decide sobre SMOTE: dataset balanceado, então NÃO usa SMOTE
    y_counts = y_train.value_counts(normalize=True).to_dict()
    logger.info(f"Balanço do alvo no treino (normalizado): {y_counts}. SMOTE NÃO será usado neste dataset.")

    # Treina e avalia
    results = train_and_evaluate(X_train, X_test, y_train, y_test, preproc, args.output_dir, args.random_state, args.n_jobs)

    # Tuning
    if args.run_tuning:
        gs = tune_decision_tree(X_train, y_train, preproc, args.output_dir, cv_splits=5, n_jobs=args.n_jobs, random_state=args.random_state)
        # avalia melhor modelo no teste
        best_pipe = gs.best_estimator_
        if hasattr(best_pipe, "predict_proba"):
            y_proba = best_pipe.predict_proba(X_test)[:,1]
        else:
            y_proba = best_pipe.predict(X_test)
        auc = roc_auc_score(y_test, y_proba)
        logger.info(f"Melhor AUC no teste para DecisionTree após tuning: {auc:.4f}")

    logger.info("Pipeline finalizada. Verifique o diretório de saída para os artefatos.")


if __name__ == "__main__":
    main()
