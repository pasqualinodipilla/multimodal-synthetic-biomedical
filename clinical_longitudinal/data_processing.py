import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH = Path("data/parkinson/parkinsons_updrs.data") #raw data path
OUT_PATH = Path("data/parkinson/processed/parkinson_T6_motor_UPDRS.npz") #preprocessed data path
SEQ_LEN = 6 #finestre temporali da 6 visite
TEST_SIZE = 0.2 #20% pazienti nel test
SEED = 42

PATIENT_COL = "subject#" #identificativo paziente
TIME_COL = "age" #utilizziamo questa colonna per ordinare le visite nel tempo (age come proxy del tempo)
TARGET_COL = "motor_UPDRS" #variabile da predire 

#leggo file, pulisco nomi colonne dagli spazi, restituisco df
def load_raw():
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns] 
    return df


def build_sequences(df, feature_cols):
    #ordino tutto per (paziente, tempo) -> ogni gruppo paziente è in ordine cronologico
    df = df.sort_values([PATIENT_COL, TIME_COL])

    #liste dove accumulo finestre, targt e id paziente
    X_list, y_list, pid_list = [], [], []

    #separo i dati per paziente con groupby, g è sotto-tabella di un paziente -> se un paziente ha meno di 6 visite
    #non possiamo fare nessuna sequenza -> lo saltiamo
    for pid, g in df.groupby(PATIENT_COL):
        g = g.reset_index(drop=True)
        if len(g) < SEQ_LEN:
            continue

        Xg = g[feature_cols].to_numpy(dtype=np.float32) #matrice delle feature del paziente (visite x features)
        yg = g[TARGET_COL].to_numpy(dtype=np.float32) #vettore target del paziente (visite)

        # sliding windows
        # prendo le visite [start : start+6]
        # target = ultimo valore della finestra (end-1)
        #memorizzo anche l'id del paziente per ogni sequenza tipo: visite v1..v8
        #creo : [v1,...,v6] target v6, [v2,...,v7] target v7, [v3,...,v8] target v8
        for start in range(0, len(g) - SEQ_LEN + 1):
            end = start + SEQ_LEN
            X_list.append(Xg[start:end])
            y_list.append(yg[end - 1])  
            pid_list.append(pid)

    #trasformo le liste in array:
    X = np.stack(X_list, axis=0)          # (N_seq, T=6, D_features)
    y = np.array(y_list, dtype=np.float32) # (N_seq,)
    pid = np.array(pid_list) #(N_seq,) id paziente di ogni sequenza

    return X, y, pid

#invece di dividere le sequenze a caso, prendiamo gli id pazienti unici
def split_by_patient(X, y, pid):
    unique_pids = np.unique(pid)
    #pazienti train e pazienti test
    train_pids, test_pids = train_test_split(
        unique_pids, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )

    #maschera booleana: quali sequenze appartengono a pazienti train/test
    train_mask = np.isin(pid, train_pids)
    test_mask = np.isin(pid, test_pids)
    #restituiamo le sequenze train/test senza mescolare pazienti
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask], pid[train_mask], pid[test_mask]

#normalizzazione (fit solo su train)
def scale(X_train, X_test):
    n_train, T, D = X_train.shape
    n_test = X_test.shape[0]

    #StandardScaler vuole (N,D) quindi appiattiamo il tempo: (N_seq, T, D) -> (N_seq*T, D)
    scaler = StandardScaler()
    #fit solo sul train -> evitiamo leakage statistico
    scaler.fit(X_train.reshape(n_train * T, D))

    #trasformiamo train e test usando lo scaler già fit poi rimettiamo shape (N_seq,T,D)
    X_train_s = scaler.transform(X_train.reshape(n_train * T, D)).reshape(n_train, T, D)
    X_test_s = scaler.transform(X_test.reshape(n_test * T, D)).reshape(n_test, T, D)

    return X_train_s.astype(np.float32), X_test_s.astype(np.float32)

#orchestation + salvataggio
def main():
    df = load_raw() #carico dataset

    #scegliamo feature = tutte le colonne tranne id paziente e target
    exclude = {PATIENT_COL, TARGET_COL}
    feature_cols = [c for c in df.columns if c not in exclude]

    #costruisco sequenze, split per paziente e normalizzo
    X, y, pid = build_sequences(df, feature_cols)
    X_tr, y_tr, X_te, y_te, pid_tr, pid_te = split_by_patient(X, y, pid)
    X_tr, X_te = scale(X_tr, X_te)

    #creo cartella destinazione se manca, salvo tutto in un file .npz compresso (così baseline_gru.py e
    # il generatore useranno questo file senza rifare preprocessing ogni volta):
    #- X_train, y_train, X_test, y_test
    #- pid_train, pid_test
    #- feature_cols, target, seq_len
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        pid_train=pid_tr,
        pid_test=pid_te,
        feature_cols=np.array(feature_cols, dtype=object),
        target_col=TARGET_COL,
        seq_len=SEQ_LEN
    )

    #stampo shape e numero pazienti per sanity check
    print("Saved:", OUT_PATH)
    print("X_train:", X_tr.shape, "X_test:", X_te.shape)
    print("y_train:", y_tr.shape, "y_test:", y_te.shape)
    print("Train patients:", len(np.unique(pid_tr)), "Test patients:", len(np.unique(pid_te)))


if __name__ == "__main__":
    main()
