function H_fd = chained_derivatives_grad(x, k, mode)
% FINDIFF_HESS_TRIDIAG Calcola l'Hessiano sparso (tridiagonale) usando FD
%
% H = findiff_Hess_tridiag(grad_fun, x, k, mode)
%
% INpUT:
%   grad_fun : function handle che restituisce il gradiente esatto (colonna)
%   x        : punto attuale (vettore colonna)
%   k        : esponente per il passo (h = 10^-k)
%   mode   : 'fixed' per h fisso, 'adaptive' per h dipendente da x
%
% OUTpUT:
%   H        : matrice sparsa tridiagonale dell'Hessiano approssimato

    n = length(x);
    base_h = 10^(-k);

    [~, g_exact] = chained_freudenstein_roth(x);
    
    if strcmp(mode, 'fixed')
        % h = 10^-k (Foto 1 e 2)
        h = ones(n, 1) * base_h;
    else
        % h_i = 10^-k * |x_i| (Foto 1 e 2)
        h = base_h * max(abs(x), 1e-8);
        % protezione: se x_i è zero, h_i sarebbe zero (divisione nulla).
        % Usiamo un valore minimo basato sulla precisione di macchina.
        h(h < eps) = base_h; 
    end

    H_diag0 = zeros(n, 1);
    H_diag1 = zeros(n-1, 1);

    % Definizione dei 3 gruppi (colori) per la struttura tridiagonale
    % Gruppo 1: 1, 4, 7...
    % Gruppo 2: 2, 5, 8...
    % Gruppo 3: 3, 6, 9...
    for group = 1:3
        % Indici delle colonne appartenenti a questo gruppo
        idx = group:3:n;
        
        if isempty(idx)
            continue;
        end
        
        % Vettore di perturbazione simultanea per il gruppo
        h_vec = zeros(n, 1);
        h_vec(idx) = h(idx);
        
        % Calcolo del gradiente nel punto perturbato
        [~, g_plus] = chained_freudenstein_roth(x + h_vec);
        
        % Differenza dei gradienti
        y = g_plus - g_exact;
        
        % Distribuzione dei risultati nei vettori diagonali
        for j = idx
            % Diagonale principale: H(j,j)
            H_diag0(j) = y(j) / h(j);
            
            % Sopradiagonale: H(j, j+1) -> salvata in H_diag1(j)
            % Nota: Per simmetria H(j+1, j) = H(j, j+1)
            if j < n
                H_diag1(j) = y(j+1) / h(j);
            end
        end
    end
    
    % 4. Costruzione della matrice tridiagonale sparsa
    % spdiags accetta matrici dove le colonne sono le diagonali.
    % Le diagonali sono [-1, 0, 1] relative alla principale.
    % Padding: per la diagonale -1 aggiungiamo uno 0 alla fine, per la +1 all'inizio.
    H_fd = spdiags([[H_diag1; 0], H_diag0, [0; H_diag1]], -1:1, n, n);
    
    % 5. Simmetrizzazione
    % Le differenze finite possono introdurre piccoli errori di asimmetria numerica
    H_fd = (H_fd + H_fd') / 2;

end