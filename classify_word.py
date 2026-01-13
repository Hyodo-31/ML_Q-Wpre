import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample
import numpy as np
import sys
import csv
from sklearn.svm import SVC

def load_and_preprocess_exclude3(csv_file, feature_set, label_column='hesitate', iteration=None):
    print(f"\n=== [Exclude3] データセット作成: {feature_set['name']}, iteration={iteration} ===")
    
    try:
        data = pd.read_csv(csv_file)
        data.columns = data.columns.str.strip().str.lower()
    except Exception as e:
        print(f"CSVの読み込み中にエラーが発生しました: {e}")
        sys.exit(1)

    # 特徴量列の選択
    if feature_set['type'] == 'all_after_understand':
        start_col = 'understand'
        if start_col not in data.columns:
            print(f"エラー: '{start_col}' 列が存在しません。")
            sys.exit(1)
        start_idx = data.columns.get_loc(start_col)
        feature_columns = data.columns[start_idx:].tolist()
    elif feature_set['type'] == 'specific':
        feature_columns = feature_set['columns']
    else:
        print("エラー: 不明な特徴量セットタイプです。")
        sys.exit(1)
    
    exclude_cols = feature_set.get('exclude_columns', [])
    feature_columns = [c for c in feature_columns if c not in exclude_cols]

    # 特徴量の存在チェック
    missing_cols = [c for c in feature_columns if c not in data.columns]
    if missing_cols:
        print(f"エラー: 指定された特徴量が見つかりません: {missing_cols}")
        print("CSVに含まれる列名:", data.columns.tolist())
        sys.exit(1)

    # ラベルの前処理
    data[label_column] = data[label_column].astype(str).str.lower()
    
    def map_label(val):
        if val in ['true', '2']: return 1
        elif val in ['false', '4']: return 0
        elif val == '3': return -1 # 除外
        return -1

    data['label_mapped'] = data[label_column].apply(map_label)
    data = data[data['label_mapped'] != -1]
    data['label_mapped'] = data['label_mapped'].astype(int)

    X_all = data[feature_columns]
    y_all = data['label_mapped']

    # 1:1 ダウンサンプリング
    df = pd.concat([X_all, y_all], axis=1)
    df_true  = df[df['label_mapped'] == 1]
    df_false = df[df['label_mapped'] == 0]
    
    if len(df_true) == 0 or len(df_false) == 0:
        print("警告: どちらかのクラスのデータが0件です。")
        return pd.DataFrame(), pd.Series(), feature_columns

    min_count= min(len(df_true), len(df_false))
    rs = 42 if iteration is None else 42 + iteration

    df_true_down  = resample(df_true,  replace=False, n_samples=min_count, random_state=rs)
    df_false_down = resample(df_false, replace=False, n_samples=min_count, random_state=rs)
    df_balanced   = pd.concat([df_true_down, df_false_down])

    X_balanced = df_balanced[feature_columns]
    y_balanced = df_balanced['label_mapped'].astype(int)

    return X_balanced, y_balanced, feature_columns


def load_and_preprocess_merge3and4(csv_file, feature_set, label_column='hesitate', iteration=None):
    print(f"\n=== [Merge3and4] データセット作成: {feature_set['name']}, iteration={iteration} ===")

    try:
        data = pd.read_csv(csv_file)
        data.columns = data.columns.str.strip().str.lower()
    except Exception as e:
        print(f"CSVの読み込み中にエラーが発生しました: {e}")
        sys.exit(1)

    feature_columns = feature_set['columns']
    
    missing_cols = [c for c in feature_columns if c not in data.columns]
    if missing_cols:
        print(f"エラー: 指定された特徴量が見つかりません: {missing_cols}")
        sys.exit(1)

    data[label_column] = data[label_column].astype(str).str.lower()
    
    def map_label(val):
        if val in ['true', '2']: return 1
        elif val in ['false', '4', '3']: return 0 # 3もFalse扱い
        return -1

    data['label_mapped'] = data[label_column].apply(map_label)
    data = data[data['label_mapped'] != -1]
    data['label_mapped'] = data['label_mapped'].astype(int)

    X_all = data[feature_columns]
    y_all = data['label_mapped']

    df = pd.concat([X_all, y_all], axis=1)
    df_true  = df[df['label_mapped'] == 1]
    df_false = df[df['label_mapped'] == 0]
    
    if len(df_true) == 0 or len(df_false) == 0:
        print("警告: どちらかのクラスのデータが0件です。")
        return pd.DataFrame(), pd.Series(), feature_columns

    min_count= min(len(df_true), len(df_false))
    rs = 42 if iteration is None else 42 + iteration

    df_true_down  = resample(df_true,  replace=False, n_samples=min_count, random_state=rs)
    df_false_down = resample(df_false, replace=False, n_samples=min_count, random_state=rs)
    df_balanced   = pd.concat([df_true_down, df_false_down])

    X_balanced = df_balanced[feature_columns]
    y_balanced = df_balanced['label_mapped'].astype(int)

    return X_balanced, y_balanced, feature_columns


def do_10fold_evaluation(X, y, iteration, feature_set_name, output_csv):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model = RandomForestClassifier(random_state=42)

    fold_prec0, fold_rec0, fold_f0 = [], [], []
    fold_prec1, fold_rec1, fold_f1_ = [], [], []

    fold_index = 1
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        prec = precision_score(y_test, y_pred, average=None, zero_division=0)
        rec  = recall_score(y_test, y_pred, average=None, zero_division=0)
        f    = f1_score(y_test, y_pred, average=None, zero_division=0)

        # 簡易的な配列処理
        p0 = prec[0] if len(prec) > 0 else 0
        r0 = rec[0] if len(rec) > 0 else 0
        f0 = f[0] if len(f) > 0 else 0
        p1 = prec[1] if len(prec) > 1 else 0
        r1 = rec[1] if len(rec) > 1 else 0
        f1 = f[1] if len(f) > 1 else 0

        fold_prec0.append(p0)
        fold_rec0.append(r0)
        fold_f0.append(f0)
        fold_prec1.append(p1)
        fold_rec1.append(r1)
        fold_f1_.append(f1)

        with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_out:
            writer = csv.writer(f_out)
            writer.writerow([feature_set_name, f"{iteration}", f"{fold_index}", "クラス0(FALSE)", p0, r0, f0])
            writer.writerow([feature_set_name, f"{iteration}", f"{fold_index}", "クラス1(TRUE)", p1, r1, f1])

        fold_index += 1
    
    return (fold_prec0, fold_rec0, fold_f0, fold_prec1, fold_rec1, fold_f1_)


def main():
    output_csv = "evaluation_results_Updated_predictedUnderstand.csv"

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["feature_set", "iteration", "fold", "class_label", "precision", "recall", "f1"])

    csv_file_name = 'wordoutputdata_hesitate_all.csv'
    csv_file = csv_file_name 

    # 特徴量セット
    feature_sets = [
        {
            'name': 'understand追加前',
            'type': 'specific',
            'columns': ['dd_time_sum', 'dd_dist_sum', 'dd_speed_min', 'dd_stop_sum',
                        'dd_uturnx_sum', 'dd_uturny_sum', 'int_time_sum'],
            'exclude_columns': []
        },
        {
            'name': 'understand追加後',
            'type': 'specific',
            'columns': ['understand', 'dd_time_sum', 'dd_dist_sum', 'dd_speed_min', 'dd_stop_sum',
                        'dd_uturnx_sum', 'dd_uturny_sum', 'int_time_sum'],
            'exclude_columns': []
        }
    ]

    n_repeat = 10

    for feature_set in feature_sets:
        print(f"\n=== 処理対象: {feature_set['name']} ===")

        # ==========================================
        # (A) Exclude3: 3を除外するパターン
        # ==========================================
        iteration_prec0_list_A, iteration_rec0_list_A, iteration_f0_list_A = [], [], []
        iteration_prec1_list_A, iteration_rec1_list_A, iteration_f1_list_A = [], [], []

        for i in range(1, n_repeat+1):
            X_bal, y_bal, selected_cols = load_and_preprocess_exclude3(
                csv_file=csv_file,
                feature_set=feature_set,
                label_column='hesitate',
                iteration=i
            )
            if X_bal.empty: continue

            (fold_prec0, fold_rec0, fold_f0,
             fold_prec1, fold_rec1, fold_f1_) = do_10fold_evaluation(
                X_bal, y_bal, i, feature_set['name'] + " (excl3)", output_csv
            )

            # 各iterationの平均
            avg_p0 = np.mean(fold_prec0)
            avg_r0 = np.mean(fold_rec0)
            avg_f0 = np.mean(fold_f0)
            avg_p1 = np.mean(fold_prec1)
            avg_r1 = np.mean(fold_rec1)
            avg_f1_ = np.mean(fold_f1_)

            iteration_prec0_list_A.append(avg_p0)
            iteration_rec0_list_A.append(avg_r0)
            iteration_f0_list_A.append(avg_f0)
            iteration_prec1_list_A.append(avg_p1)
            iteration_rec1_list_A.append(avg_r1)
            iteration_f1_list_A.append(avg_f1_)

            # iterationごとの平均を書き込み
            with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_out:
                writer = csv.writer(f_out)
                writer.writerow([feature_set['name'] + " (excl3)", f"iteration_{i}", "avg_of_10folds", "クラス0(FALSE)", avg_p0, avg_r0, avg_f0])
                writer.writerow([feature_set['name'] + " (excl3)", f"iteration_{i}", "avg_of_10folds", "クラス1(TRUE)", avg_p1, avg_r1, avg_f1_])

        # ★追加: Exclude3 の final_average を書き込み
        if iteration_prec0_list_A:
            final_p0_A = np.mean(iteration_prec0_list_A)
            final_r0_A = np.mean(iteration_rec0_list_A)
            final_f0_A = np.mean(iteration_f0_list_A)
            final_p1_A = np.mean(iteration_prec1_list_A)
            final_r1_A = np.mean(iteration_rec1_list_A)
            final_f1_A = np.mean(iteration_f1_list_A)

            print(f"  [Exclude3] Final Average: クラス1 F1 = {final_f1_A:.4f}")

            with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_out:
                writer = csv.writer(f_out)
                writer.writerow([feature_set['name'] + " (excl3)", "final_average", "-", "クラス0(FALSE)", final_p0_A, final_r0_A, final_f0_A])
                writer.writerow([feature_set['name'] + " (excl3)", "final_average", "-", "クラス1(TRUE)", final_p1_A, final_r1_A, final_f1_A])


        # ==========================================
        # (B) Merge3and4: 3と4を統合するパターン
        # ==========================================
        iteration_prec0_list_B, iteration_rec0_list_B, iteration_f0_list_B = [], [], []
        iteration_prec1_list_B, iteration_rec1_list_B, iteration_f1_list_B = [], [], []

        for i in range(1, n_repeat+1):
            X_bal, y_bal, selected_cols = load_and_preprocess_merge3and4(
                csv_file=csv_file,
                feature_set=feature_set,
                label_column='hesitate',
                iteration=i
            )
            if X_bal.empty: continue

            (fold_prec0, fold_rec0, fold_f0,
             fold_prec1, fold_rec1, fold_f1_) = do_10fold_evaluation(
                X_bal, y_bal, i, feature_set['name'] + " (merge3_4)", output_csv
            )

            # 各iterationの平均
            avg_p0 = np.mean(fold_prec0)
            avg_r0 = np.mean(fold_rec0)
            avg_f0 = np.mean(fold_f0)
            avg_p1 = np.mean(fold_prec1)
            avg_r1 = np.mean(fold_rec1)
            avg_f1_ = np.mean(fold_f1_)

            iteration_prec0_list_B.append(avg_p0)
            iteration_rec0_list_B.append(avg_r0)
            iteration_f0_list_B.append(avg_f0)
            iteration_prec1_list_B.append(avg_p1)
            iteration_rec1_list_B.append(avg_r1)
            iteration_f1_list_B.append(avg_f1_)

            # iterationごとの平均を書き込み
            with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_out:
                writer = csv.writer(f_out)
                writer.writerow([feature_set['name'] + " (merge3_4)", f"iteration_{i}", "avg_of_10folds", "クラス0(FALSE)", avg_p0, avg_r0, avg_f0])
                writer.writerow([feature_set['name'] + " (merge3_4)", f"iteration_{i}", "avg_of_10folds", "クラス1(TRUE)", avg_p1, avg_r1, avg_f1_])

        # ★追加: Merge3and4 の final_average を書き込み
        if iteration_prec0_list_B:
            final_p0_B = np.mean(iteration_prec0_list_B)
            final_r0_B = np.mean(iteration_rec0_list_B)
            final_f0_B = np.mean(iteration_f0_list_B)
            final_p1_B = np.mean(iteration_prec1_list_B)
            final_r1_B = np.mean(iteration_rec1_list_B)
            final_f1_B = np.mean(iteration_f1_list_B)

            print(f"  [Merge3and4] Final Average: クラス1 F1 = {final_f1_B:.4f}")

            with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_out:
                writer = csv.writer(f_out)
                writer.writerow([feature_set['name'] + " (merge3_4)", "final_average", "-", "クラス0(FALSE)", final_p0_B, final_r0_B, final_f0_B])
                writer.writerow([feature_set['name'] + " (merge3_4)", "final_average", "-", "クラス1(TRUE)", final_p1_B, final_r1_B, final_f1_B])


if __name__ == "__main__":
    main()