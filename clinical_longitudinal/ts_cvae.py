import torch  
import torch.nn as nn  
import torch.nn.functional as F  

#implementiamo un cvae (conditional variational autoencoder) per serie temporali con gru

#l'encoder prendere in input una sequenza x (serie temporale) e una condizione c (info extra) e produce 2 vettori
# mu = media della distribuzione latente
# logvar = log(varianza) della distribuzione latente
class Encoder(nn.Module):
    #costruttore: input_dim = dim delle feature di x (D), cond_dim = dim della condizione c
    # hidden_dim = dim dello stato interno della gru, latent_dim = dim dello spazio latente z  
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim):
        super().__init__()  #inizializzazione
        #creo una gru che riceve in input, ad ogni istante temporale, un vettore di dim input_dim + cond_dim
        self.gru = nn.GRU(input_dim+cond_dim, hidden_dim, batch_first=True) #batch_first -> sequenza in formato (B,T,D)
        #due layer lineari che trasformano l'ultimo stato nascosto della gru (dimensione hidden_dim) in:
        # mu (dimensione latent_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        #logvar (dimensione latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    #x è una sequenza: batch B, lunghezza T, feature D
    #c è un vettore per ogni elemento del batch
    def forward(self, x, c):
        #leggo batch size B e lunghezza sequenza T
        B, T, _ = x.shape
        #con c.unsqueeze(1) aggiungo una dimensione : da (B, cond_dim) a (B, 1, cond_dim)
        #repeat(1, T, 1) ripete lungo il tempo (B,T,cond_dim)
        #praticamente copio la condizione c su tutti gli istnati temporali
        c_seq = c.unsqueeze(1).repeat(1, T, 1) 
        #concateno lungo l'ultima dimensione (le feature): ottengo l'input per la gru
        #ora ogni time-step contiene [feature di x | conditio c]
        xc = torch.cat([x, c_seq], dim=-1)
        #passo la sequenza nella gru, out ha shape (B,T,hidden_dim) = output a ogni time-step
        #_ sarebbe l'ultimo hidden state ma non lo usiamo
        out, _ = self.gru(xc)
        #prendo l'output dell'ultimo time-step (-1)
        #h_last ha shape (B, hidden_dim)
        h_last = out[:, -1,:]
        #produco due vettori mu: (B, latent_dim) e logvar: (B, latent_dim)
        return self.mu(h_last), self.logvar(h_last)


#il decoder prende un campione latente z, la condizione c e ricostruisce una squenza x_hat lunga seq_len
class Decoder(nn.Module):  # Decoder (spazio latente -> sequenza ricostruita)
    def __init__(self, latent_dim, cond_dim, hidden_dim, output_dim, seq_len):
        super().__init__()
        #seq_len = T, lunghezza della sequenza da generare
        self.seq_len = seq_len 
        #la gru del decoder riceve ad ogni istante un vettore di dimensione latent_dim + cond_dim
        self.gru = nn.GRU(latent_dim+cond_dim, hidden_dim, batch_first=True)
        #trasforma l'output GRU (hidden_dim) nelle feature finali (output_dim, che qui sara input_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, z,c):
        #z: (B,Z), c: (B,cond_dim)
        B = z.shape[0] #batch size
        zc = torch.cat([z,c], dim=-1) #concatena z e c -> un solo vettore
        zc_seq = zc.unsqueeze(1).repeat(1, self.seq_len, 1) #lo stende nel tempo -> stessa info per tutti i time step
        #quindi il decoder riceve una sequenza costante (non autoregressiva)

        #produce out: (B,T,hidden_dim)
        out, _ = self.gru(zc_seq)
        #layer lineare su ogni time-step -> x_hat con shape (B,T,output_dim)
        return self.output(out)


class TSCVAE(nn.Module):  # Time Series Conditional Variational AutoEncoder -> unisce encoder + decoder
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim, seq_len):
        super().__init__()
        # Creo encoder
        self.encoder = Encoder(input_dim, cond_dim, hidden_dim, latent_dim)
        # Creo decoder
        self.decoder = Decoder(latent_dim, cond_dim, hidden_dim, input_dim, seq_len)
        #il decoder ricostruisce input_dim perchè vuole ricreare x


    def forward(self, x,c):
        # x: (B, T, D)
        
        mu, logvar = self.encoder(x,c) #encoder -> (mu, logvar) entrambi (B, latent_dim)  
        std = torch.exp(0.5 * logvar) #logvar = log(sigma^2), quindi sigma = exp(0.5*logvar)
        eps = torch.randn_like(std)  #rumore gaussiano standard (N(0,1)) con la stessa shape di std
        z = mu + eps * std  #campioni z che vanno come N(mu,sigma^2) ma in modo differenziabile
        x_hat = self.decoder(z,c)  #decoder -> ricostruzione (B,T,input_dim)
        return x_hat, mu, logvar #mi serve mu e logvar per calcolare la KL loss


def cvae_loss(x, x_hat, mu, logvar):
    # Reconstruction loss: quanto bene ricostruisco l'input cioè quanto x_hat è vicino a x
    #mse media su tutti gli elementi
    recon = F.mse_loss(x_hat, x, reduction="mean")
    # Mean Squared Error media su batch e dimensioni
    
    # KL divergence: quanto la distribuzione latente si allontana da N(0, I)
    #cioè KL divergence tra la distribuzione appresa q(z|x,c) = N(mu,sigma^2) e la prior p(z)=N(0,1)
    #spinge mu verso 0 e sigma verso 1 (regolarizzazione)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Loss totale VAE: totale = ricostruzione + regolarizzzazione
    return recon + kl
