from pathlib import Path #per gestire percorsi file
import numpy as np #per array e calcolo metriche (mse, mae, r2)
import torch
from torch import nn #moduli neural network (gru, linear ...)
from torch.utils.data import Dataset, DataLoader #per passare i dati al modello a batch

#carico sequenze reali (6x20) -> alleno una GRU per predire motor_UPDRS -> misuro l'errore su pazienti mai visti
#-> salvo il modello migliore

NPZ_PATH = Path("data/parkinson/processed/parkinson_T6_motor_UPDRS.npz") #percorso del dataset preprocessato

SEED=42
BATCH_SIZE = 128 #quante sequenze per aggiornamento
EPOCHS = 30 #quante "passate" complete sul train
LR = 1e-3 #velocità con cui il modello impara (learning rate)

#iperparametri della GRU
HIDDEN_SIZE = 64 #dimensione memoria interna
NUM_LAYERS = 1 #numero GRU impilate
DROPOUT = 0.0 #regolarizzazione (0 perchè 1 layer)

#con il mio mac posso usar MPS (accelerazione GPU-like) altrimenti cpu
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

#rendiamo tutto riproducibile: inizializzazione pesi, shuffle, ecc. siano ripetibili
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

#converto gli array in tensori PyTorch
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        #prendo X e y (numpy) e li converto in tensori torch, float32 standard per reti neurali
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    #dico a pytorch quanti esempi ci sono
    def __len__(self):
        return self.X.shape[0]
    #restituisco un singolo esempio (sequenza, target)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
#modello: GRU+testa lineare    
class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        #definiamo un modello pytorch personalizzato, con init inizializziamo la classe base
        self.gru = nn.GRU( #rete ricorrente che legge una sequenza nel tempo
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first = True, # significa che l'input è (B,T,D), 
            #con B=batch_size, T=6 e D=20
        )
        #dopo la GRU otteniamo un vettore di dimensione hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size), #stabilizza
            nn.Linear(hidden_size,1), #produce il valore predetto (motor_UPDRS)
        )

    #Come si calcola una predizione
    def forward(self,x):
        #x: sequenza (B,T=6,D=20)
        out, _ = self.gru(x) #out diventa (B,6,hidden), _ è lo stato nascosto finale che non usiamo qui
        last = out[:,-1,:] #(B,H) -> prendiamo l'output dell'ultimo time-step (t=6) e otteniamo (B,hidden)
        yhat = self.head(last).squeeze(-1) # (B,) -> head produce (B,1), squeeze -> (B,)
        return yhat #restituzione predizione
    
#valutazione: calcolo metriche su test    
@torch.no_grad() #non calcola gradienti (più veloce, meno memoria)
def evaluate(model, loader): #usiamo loader che scorre i dati a batch
    model.eval() #mette il modello in modalità valutazione (no dropout ecc.)
    #accumulo tutte le y vere e y predette
    #calcolo: mse (errore quadratico medio), mae (errore assoluto medio), r2 (quanto spiega la var rispetto al baseline "media")
    ys, yhs = [], [] 
    for X, y in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        yhat = model(X)
        ys.append(y.detach().cpu().numpy())
        yhs.append(yhat.detach().cpu().numpy())

    y = np.concatenate(ys)
    yhat = np.concatenate(yhs)

    mse = float(np.mean((yhat-y)**2))
    mae = float(np.mean(np.abs(yhat-y)))
    denom = np.sum((y-y.mean())**2)+1e-12
    r2 = float(1.0-np.sum((yhat-y)**2)/denom)
    return mse, mae, r2

#orchestration del training
def main():
    set_seed(SEED) #fisso seed

    #se non ho creato l'npz con preprocessing mi fermo
    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"Non trovo {NPZ_PATH}. Prima esegui: python clinical_longitudinal/data_processing.py"
        )
    
    #carico il dataset preprocessato
    data = np.load(NPZ_PATH, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print("Loaded:", NPZ_PATH)
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("Device:", DEVICE)

    #trasformo in dataset pytorch
    train_ds = SeqDataset(X_train, y_train)
    test_ds = SeqDataset(X_test, y_test)

    #dataloader crea batch, shuffle=True solo nel train
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    #creo modello e ottimizzatore
    #input_dim = 20 feature, sposto modello su device (mps o cpu)
    input_dim = X_train.shape[-1]
    model = GRURegressor(input_dim, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)

    #adam = ottimizzator standard, loss=mse
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_mse = float("inf")
    best_state= None

    #training loop: per ogni epoca ->
    #passa su tutti i batch train, calcola predizione, calcola loss, fa backprop
    #e aggiorna pesi e poi valuta su test
    #e salva il modello migliore (quello con test mse più basso)
    for epoch in range(1, EPOCHS+1):
        model.train()
        running = 0.0

        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            yhat = model(X)
            loss = loss_fn(yhat,y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item() * X.size(0)

        train_mse = running / len(train_ds)
        test_mse, test_mae, test_r2 = evaluate(model, test_loader)

        if test_mse < best_mse:
            best_mse = test_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | train_MSE={train_mse:.4f} | "
            f"test_MSE={test_mse:.4f} test_MAE={test_mae:.4f} test_R2={test_r2:.4f}"
        )

    #salvo i pesi del modello migliore su disco
    out_dir = Path("clinical_longitudinal/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "baseline_gru_best.pt"
    torch.save(best_state, ckpt_path)

    print("\nSaved best model:", ckpt_path)
    print("Best test MSE:", best_mse)

if __name__ == "__main__":
    main()