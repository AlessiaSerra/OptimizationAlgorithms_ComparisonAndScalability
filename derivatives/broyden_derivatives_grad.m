function [g_exact, H_fd] = broyden_derivatives_grad(x, k, mode)
    % x: punto in cui calcolare le derivate (x_hat nella foto)
    % k: esponente per l'incremento (4, 8, 12)
    % mode: 'fixed' per h costante, 'variable' per h_i proporzionale a |x_i|
    
    n = length(x);
    base_h = 10^(-k);
    
    % 1. Calcolo del Gradiente Esatto (come richiesto dalla seconda foto)
    % Utilizziamo la funzione analitica fornita nel primo messaggio
    [~, g_exact] = broyden_tridiagonal(x);
    
    % 2. Definizione dei vettori di incremento h (secondo le formule nelle foto)
    if strcmp(mode, 'fixed')
        % h = 10^-k (Foto 1 e 2)
        h = ones(n, 1) * base_h;
    else
        % h_i = 10^-k * |x_i| (Foto 1 e 2)
        h = base_h * max(abs(x), 1e-8);
        % Protezione: se x_i è zero, h_i sarebbe zero (divisione nulla).
        % Usiamo un valore minimo basato sulla precisione di macchina.
        h(h < eps) = base_h; 
    end
    
    % 3. Approssimazione dell'Hessiana (Differenze finite sul Gradiente)
    % Sfruttiamo la struttura pentadiagonale: calcoliamo solo le interazioni locali.
    H_diag0 = zeros(n, 1);
    H_diag1 = zeros(n-1, 1);
    H_diag2 = zeros(n-2, 1);
    
    % Invece di un ciclo da 1 a n, facciamo un ciclo da 1 a 5 (i "colori")
    for group = 1:5
        % Identifichiamo gli indici che appartengono a questo gruppo (es. 1, 6, 11...)
        idx = group:5:n;
        % Creiamo un UNICO vettore di perturbazione per tutto il gruppo
        h_vec = zeros(n, 1);
        h_vec(idx) = h(idx);
        
        % UNICA valutazione del gradiente per 1/5 delle colonne totali
        [~, g_plus] = broyden_tridiagonal(x + h_vec);
        
        % Differenza vettoriale
        y = g_plus - g_exact;
        
        % Ora distribuiamo i risultati nei vettori diagonali
        % Poiché le bande non si sovrappongono, y(j) corrisponde alla colonna j
        for j = idx
            % Diagonale principale
            H_diag0(j) = y(j) / h(j);
            
            % Sopradiagonale 1
            if j <= n-1
                H_diag1(j) = y(j+1) / h(j);
            end
            
            % Sopradiagonale 2
            if j <= n-2
                H_diag2(j) = y(j+2) / h(j);
            end
        end
    end
    
    % 4. Costruzione della matrice pentadiagonale sparsa
    % spdiags distribuisce i vettori sulle diagonali [-2, -1, 0, 1, 2]
    H_fd = spdiags([[H_diag2;0;0], [H_diag1;0], H_diag0, [0;H_diag1], [0;0;H_diag2]], -2:2, n, n);

    %Per garanzia numerica (errori di arrotondamento)
    H_fd = (H_fd + H_fd') / 2;
end