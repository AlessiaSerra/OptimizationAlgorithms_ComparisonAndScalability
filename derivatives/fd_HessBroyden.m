function H_fd = fd_HessBroyden(x, k, mode)
    % Calcolo ottimizzato dell'Hessiana Finite Differences O(n)
    % Versione corretta con H_vec allineato agli indici di X
    
    n = length(x);
    base_h = 10^(-k);
    
    % 1. Definizione vettore h
    if strcmp(mode, 'fixed')
        h = ones(n, 1) * base_h;
    else
        h = base_h * max(abs(x), 1e-8);
    end
    h = sqrt(h); % Step per Hessiana
    
    % 2. Funzioni Locali (Definizione di Broyden)
    % f_k = (3 - 2*x_k)*x_k - x_{k-1} - 2*x_{k+1} + 1
    calc_fk = @(xp, xc, xn) (3 - 2*xc).*xc - xp - 2*xn + 1;
    calc_E  = @(xp, xc, xn) 0.5 * calc_fk(xp, xc, xn).^2;
    
    % 3. Creazione vettori con Padding e Shift
    % X_pad = [0, x_1, ..., x_n, 0]
    X = [0; x; 0];
    % H_vec = [0, h_1, ..., h_n, 0] -> Cruciale per associare il passo h_i corretto a x_i
    H_vec = [0; h; 0];
    
    % Pre-calcolo energia base
    % xp=X(1:n), xc=X(2:n+1), xn=X(3:n+2)
    E0 = calc_E(X(1:n), X(2:n+1), X(3:n+2));
    
    % =========================================================
    % 1. Diagonale Principale (i, i)
    % =========================================================
    
    % Forward (x + h)
    % Usiamo H_vec con gli stessi indici di X per garantire h corretto
    Ep_prev = calc_E(X(1:n), X(2:n+1), X(3:n+2) + H_vec(3:n+2)); % Perturbo x_{k+1} con h_{k+1}
    Ep_curr = calc_E(X(1:n), X(2:n+1) + H_vec(2:n+1), X(3:n+2)); % Perturbo x_k con h_k
    Ep_next = calc_E(X(1:n) + H_vec(1:n), X(2:n+1), X(3:n+2));   % Perturbo x_{k-1} con h_{k-1}
    
    % Backward (x - h)
    Em_prev = calc_E(X(1:n), X(2:n+1), X(3:n+2) - H_vec(3:n+2));
    Em_curr = calc_E(X(1:n), X(2:n+1) - H_vec(2:n+1), X(3:n+2));
    Em_next = calc_E(X(1:n) - H_vec(1:n), X(2:n+1), X(3:n+2));
    
    % Central Difference locale: (E(x+h) - 2E(x) + E(x-h)) / h^2
    % Attenzione: dividiamo per il passo della variabile perturbata!
    D_prev = (Ep_prev - 2*E0 + Em_prev) ./ (H_vec(3:n+2).^2);
    D_curr = (Ep_curr - 2*E0 + Em_curr) ./ (H_vec(2:n+1).^2);
    D_next = (Ep_next - 2*E0 + Em_next) ./ (H_vec(1:n).^2);
    
    % Assemblaggio Diagonale Principale (somma dei contributi shiftati)
    % Indice i riceve contributi da:
    % - Termine i (dove x_i è Curr)
    % - Termine i-1 (dove x_i è Next/xn) -> shiftato in avanti
    % - Termine i+1 (dove x_i è Prev/xp) -> shiftato indietro
    diag0 = ([0; D_prev(1:n-1)] + D_curr + [D_next(2:n); 0]);
    
    
    % =========================================================
    % 2. Prima Sopradiagonale (i, i+1)
    % =========================================================
    % Interazione tra x_i e x_{i+1}. Avviene in due termini f_k:
    % A) Termine k=i   : x_i è Curr, x_{i+1} è Next
    % B) Termine k=i+1 : x_i è Prev, x_{i+1} è Curr
    
    % Denominatore: h_i * h_{i+1}
    hh_cross1 = H_vec(2:n) .* H_vec(3:n+1);
    
    % Contributo A (Termine i): Perturbo Curr(i) e Next(i+1)
    % Range i=1..n-1. Curr=X(2:n), Next=X(3:n+1)
    E_crossA = calc_E(X(1:n-1), ...
                      X(2:n) + H_vec(2:n), ...      % x_i + h_i
                      X(3:n+1) + H_vec(3:n+1));     % x_{i+1} + h_{i+1}
                      
    % Sottrazioni: E(++h) - E(+hi) - E(+hj) + E0
    % Nota: Ep_curr(1:n-1) contiene E(+h_i) per questo termine
    % Nota: Ep_prev(1:n-1) contiene E(+h_{i+1}) per questo termine (x_{i+1} è next)
    numA = E_crossA - Ep_curr(1:n-1) - Ep_prev(1:n-1) + E0(1:n-1);
    
    % Contributo B (Termine i+1): Perturbo Prev(i) e Curr(i+1)
    % Range i=1..n-1. Prev=X(2:n), Curr=X(3:n+1)
    E_crossB = calc_E(X(2:n) + H_vec(2:n), ...      % x_i + h_i
                      X(3:n+1) + H_vec(3:n+1), ...  % x_{i+1} + h_{i+1}
                      X(4:n+2));
    
    % Sottrazioni:
    % Ep_next(2:n) è E(+h_i) (x_i è prev)
    % Ep_curr(2:n) è E(+h_{i+1}) (x_{i+1} è curr)
    numB = E_crossB - Ep_next(2:n) - Ep_curr(2:n) + E0(2:n);
    
    diag1 = (numA + numB) ./ hh_cross1;
    
    
    % =========================================================
    % 3. Seconda Sopradiagonale (i, i+2)
    % =========================================================
    % Interazione tra x_i e x_{i+2}. Avviene solo in un termine f_k:
    % - Termine k=i+1 : x_i è Prev, x_{i+2} è Next
    
    hh_cross2 = H_vec(2:n-1) .* H_vec(4:n+1); % h_i * h_{i+2}
    
    % Contributo Unico: Perturbo Prev(i) e Next(i+2)
    % Range i=1..n-2. Prev=X(2:n-1), Next=X(4:n+1)
    E_crossC = calc_E(X(2:n-1) + H_vec(2:n-1), ...  % x_i + h_i
                      X(3:n), ...
                      X(4:n+1) + H_vec(4:n+1));     % x_{i+2} + h_{i+2}
                      
    % Sottrazioni:
    % Ep_next(2:n-1) è E(+h_i) (x_i è prev)
    % Ep_prev(2:n-1) è E(+h_{i+2}) (x_{i+2} è next)
    numC = E_crossC - Ep_next(2:n-1) - Ep_prev(2:n-1) + E0(2:n-1);
    
    diag2 = numC ./ hh_cross2;
    
    
    % =========================================================
    % 4. Costruzione Matrice Sparsa
    % =========================================================
    H_fd = spdiags([ ...
        [diag2; 0; 0], ...           % -2
        [diag1; 0], ...              % -1
        diag0, ...                   %  0
        [0; diag1], ...              % +1
        [0; 0; diag2]], ...          % +2
        -2:2, n, n);
        
    % Simmetrizzazione
    H_fd = (H_fd + H_fd') / 2;
end