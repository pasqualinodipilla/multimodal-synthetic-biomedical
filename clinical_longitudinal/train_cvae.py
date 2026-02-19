import numpy as np  
from pathlib import Path  

import torch  
from torch.utils.data import Dataset, DataLoader 

from clinical_longitudinal.ts_cvae import TSCVAE, cvae_loss  

#parametri training
SEED = 42          
BATCH_SIZE = 128   
EPOCHS = 30        
LR = 1e-3          

HIDDEN_DIM = 64    
LATENT_DIM = 16    
COND_DIM = 1 #perchè la condizione y è scalare (un numero per paziente)

#creo la cartella dove salvare i pesi, CKPT_PATH per il file del modello migliore
OUT_DIR = Path("clinical_longitudinal/outputs")  
OUT_DIR.mkdir(parents=True, exist_ok=True)       
CKPT_PATH = OUT_DIR / "ts_cvae_best.pt"           
#creo cartella per dataset sintetico, SYNTH_PATH per salvare X_synth e y_synth
SYNTH_DIR = Path("data/parkinson/synthetic")     
SYNTH_DIR.mkdir(parents=True, exist_ok=True)     
SYNTH_PATH = SYNTH_DIR / "parkinson_T6_motor_UPDRS_tscvae_synth.npz"  

#imposto seed per rendere i risultati riproducibili, mps è la gpu apple (mac)
def set_seed(seed=SEED):
    np.random.seed(seed)        
    torch.manual_seed(seed)    
    if torch.backends.mps.is_available():   
        torch.mps.manual_seed(seed)       

#se ho mps uso mps altrimenti cpu
def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"
   

#cosa entra nel modello?
class SeqCondDataset(Dataset):
    #prendo: X con shape (N,T,D) -> N pazienti, T visite, D feature
            #y con shape (N,) -> un numero per paziente
    # lo trasformo in tensori torch
    # y.reshape(-1,1) serve per avere c shape (N,1) (perchè cond_dim=1)
    # ogni item restituisce: (X[i], c[i]) cioè : X[i] shape (T,D), c[i] shape (1,)
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  
        self.c = torch.tensor(y.reshape(-1,1), dtype=torch.float32) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.c[i]      

#Funzione per generare X sintetico dato y
@torch.no_grad() #non calcola gradienti (più veloce, meno memoria)
def sample_X_given_y(model, y, T, latent_dim, device):
    
    model.eval()  #modalità evaluation
    c = torch.tensor(y.reshape(-1,1), dtype=torch.float32, device=device) #converto y (numpy) in c tensor shape (N,1)
    z = torch.randn(len(y), latent_dim, device=device) #genero z che va come N(0,1) shape (N, latent_dim)
    x_hat = model.decoder(z,c)  #uso solo il decoder per generare x_hat shape (N,T,D)
    return x_hat.detach().cpu().numpy()  #ritorno numpy su CPU
#(qui T non lo uso perchè il decoder già sa seq_len)

#carico dati
def main():
    set_seed() #set seed             
    device = get_device()   #scelgo device
    print("Device:", device) #stampa device

    
    #carico il dataset preprocessato
    data_path = "data/parkinson/processed/parkinson_T6_motor_UPDRS.npz"  
    data = np.load(data_path, allow_pickle=True)  

    #prendo X_train -> shape (N,T,D), y_train shape (N,), feature_cols = nomi delle feature se presenti
    X_train = data["X_train"]  
    y_train = data["y_train"]    
    feature_cols = list(data["feature_cols"]) if "feature_cols" in data else None
    

    #estraggo dimensioni e stampo
    N, T, D = X_train.shape  
    print("X_train:", X_train.shape, "y_train:", y_train.shape)

    #normalizzazione di y
    #calcolo media e dev standard di y (scala originale UPDRS)
    #Trasformo y in z-score: media circa 0, std circa 1
    #salvo y_mean e y_std perchè servono poi dopo per tornare alla scala clinica
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std
    print("y normalizzata: mean=", y_train.mean(), "std=", y_train.std())


    #DataLoader: 
    #costruisco un dataset PyTorch con (X_train, y_train_normalizzato)
    #batch_size=128: training a mini-batch
    #shuffle=True mescola i pazienti ad ogni epoca

    #ogni batch darà: xb shape (B,T,D), cb shape (B,1)
    train_loader = DataLoader(
        SeqCondDataset(X_train, y_train),      
        batch_size=BATCH_SIZE,     
        shuffle=True,              
        drop_last=False,           
    )

    #modello e ottimizzatore
    #creo cvae: input_dim = D, cond_dim = 1, hidden_dim = 64, latent_dim = 16, seq_len = T
    #to(device) sposta tutto su MPS/CPU
    #adam ottimizzatore
    #best per salvare il migliore
    model = TSCVAE(input_dim=D, cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, seq_len=T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)  
    best_loss = float("inf") 

    #training loop
    for epoch in range(1, EPOCHS + 1):  
        model.train()      #modalità training 
        running = 0.0        
        #per ogni batch: 
        # sposto xb,cb sul device
        # opt.zero_grad() azzera gradienti precedenti.
        #model(xb,cb): encoder -> mu, logvar, sample z (reparameterization), decoder -> x_hat
        #cvae_loss = MSE ricostruzione + KL divergence
        #loss.backward() calcola i gradienti
        #opt.step() aggiorna pesi
        #accumuli la loss pesata per batch size

        #dopo i batch: train_loss = loss media sul dataset, se miglioe salva il modello
        for xb, cb in train_loader:    
            xb, cb = xb.to(device), cb.to(device)     

            opt.zero_grad()       
            x_hat, mu, logvar = model(xb, cb)         
            loss = cvae_loss(xb, x_hat, mu, logvar) 
            loss.backward()        
            opt.step()             

            running += loss.item() * xb.size(0)  

        train_loss = running / len(train_loader.dataset)
        

        if train_loss < best_loss:      
            best_loss = train_loss      
            torch.save(model.state_dict(), CKPT_PATH)  

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | best={best_loss:.6f}")
        

    print("\nSaved best TS-CVAE to:", CKPT_PATH)

    #generatione sintetici + denormalizzazione y
    #ricarica i pesi migliori
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    #campiona y (normalizzato) dalla distribuzione dei train labels
    #replace=True=con ripetizione
    #y_synth_norm shape (N,)
    y_synth_norm = np.random.choice(y_train, size=len(y_train), replace=True)
    #genera X_synth shape (N,T,D) condizionato su y_synth_norm
    X_synth = sample_X_given_y(model, y_synth_norm, T=T, latent_dim=LATENT_DIM, device=device)
    #riporta y sintetico alla scala UPDRS originale così che quello che salvo nel file sia clinicamente interpretabile
    y_synth = y_synth_norm * y_std + y_mean
    
    print("y_synth stats (dovrebbe essere scala UPDRS):", "min", float(y_synth.min()), "max", float(y_synth.max()),
    "mean", float(y_synth.mean()))

    #salvo in npz: X_synth, y_synth (in scala reale), metadata (feature cols, dimensioni etc.)
    np.savez(
        SYNTH_PATH,  
        X_synth=X_synth,  
        y_synth = y_synth,
        feature_cols=np.array(feature_cols, dtype=object) if feature_cols is not None else None,
        
        seq_len=T,         
        input_dim=D,       
        generator="TS-CVAE", 
        hidden_dim=HIDDEN_DIM, 
        latent_dim=LATENT_DIM, 
        seed=SEED,             
    )

    #stampo conferma
    print("Saved synthetic dataset to:", SYNTH_PATH)  
    print("X_synth:", X_synth.shape, "y_synth:", y_synth.shape)


if __name__ == "__main__":
    main()  
