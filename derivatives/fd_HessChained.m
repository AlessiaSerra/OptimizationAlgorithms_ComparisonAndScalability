function H_fd = fd_HessChained(x, k, mode)
    % Calcolo ottimizzato Hessiana Finite Differences per Chained F-R
    % Matrice Tridiagonale
    
    n = length(x);
    base_h = 10^(-k);
    
    if strcmp(mode, 'fixed')
        h = ones(n, 1) * base_h;
    else
        h = base_h * max(abs(x), 1e-8);
    end
    
    h = sqrt(h); % Step per Hessiana
    
    % Funzioni handle locali
    calc_E = @(u, v) ...
        0.5 * ( ...
            (u + v .* ((5 - v) .* v - 2) - 13).^2 + ...
            (u + v .* ((1 + v) .* v - 14) - 29).^2 ...
        );

    % Base
    E0 = calc_E(x(1:n-1), x(2:n));
    
    % === 1. Diagonale Principale (Central Difference) ===
    % H_ii dipende da come x_i influenza la somma dei quadrati.
    % x_i appare nel blocco i (come u) e nel blocco i-1 (come v)
    
    % A) Contributo dal blocco in cui x_i è "current" (u) -> indici 1:n-1
    h_curr = h(1:n-1);
    Ep_curr = calc_E(x(1:n-1) + h_curr, x(2:n));
    Em_curr = calc_E(x(1:n-1) - h_curr, x(2:n));
    D2_curr = (Ep_curr - 2*E0 + Em_curr) ./ (h_curr.^2);
    
    % B) Contributo dal blocco in cui x_i è "next" (v) -> indici 2:n
    h_next = h(2:n);
    Ep_next = calc_E(x(1:n-1), x(2:n) + h_next);
    Em_next = calc_E(x(1:n-1), x(2:n) - h_next);
    D2_next = (Ep_next - 2*E0 + Em_next) ./ (h_next.^2);
    
    % Assemblaggio diagonale principale
    % D2_curr copre indici 1..n-1, D2_next copre indici 2..n
    diag0 = zeros(n, 1);
    diag0(1:n-1) = diag0(1:n-1) + D2_curr;
    diag0(2:n)   = diag0(2:n)   + D2_next;
    
    % === 2. Sopradiagonale (Termini misti x_i, x_{i+1}) ===
    % Questo accade SOLO nel blocco i, dove abbiamo u=x_i e v=x_{i+1}
    % Calcoliamo d^2 E / (du dv) per i = 1..n-1
    
    % E(u+h, v+h)
    E_cross = calc_E(x(1:n-1) + h_curr, x(2:n) + h_next);
    
    % Formula: (E(++h) - E(+h_u) - E(+h_v) + E0) / (h_u * h_v)
    % Nota: Ep_curr è esattamente E(+h_u)
    % Nota: Ep_next è esattamente E(+h_v)
    
    num = E_cross - Ep_curr - Ep_next + E0;
    den = h_curr .* h_next;
    
    diag1 = num ./ den;
    
    % === Costruzione Matrice Sparsa ===
    H_fd = spdiags([ ...
        [0; diag1], ...    % -1 (Sotto)
        diag0, ...         %  0 (Principale)
        [diag1; 0]], ...   % +1 (Sopra)
        -1:1, n, n);
        
    % Simmetrizzazione (ridondante ma sicura)
    H_fd = (H_fd + H_fd') / 2;
    
end