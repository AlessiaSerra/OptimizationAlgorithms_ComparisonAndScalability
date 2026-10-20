function [F,g,H] = chained_freudenstein_roth(x)
    %%n = length(x); %%vecchia versione che non permette il calcolo
    %%contemporaneo di più punti

    [n, ~] = size(x); % MODIFICA 1: Usiamo size per essere sicuri di prendere le righe

    % Creiamo i due vettori "sfasati"
    % Se n=3, x_i è [x1, x2] e x_next è [x2, x3]
    % MODIFICA 2: Aggiungiamo ", :" per dire "tutte le colonne" e
    % permettere il calcolo contemporaneo di più punti
    x_i    = x(1:n-1,:);
    x_next = x(2:n,:);
    
    % Calcoliamo TUTTI i k dispari in un colpo solo
    % Formula: x_i + x_{i+1}((5 - x_{i+1})x_{i+1} - 2) - 13
    % Il punto (.) prima di * e ^ serve per dire "elemento per elemento"
    f_odd = x_i + x_next .* ((5 - x_next) .* x_next - 2) - 13;
    
    % Calcoliamo TUTTI i k pari in un colpo solo
    % Formula: x_i + x_{i+1}((1 + x_{i+1})x_{i+1} - 14) - 29
    f_even    = x_i + x_next .* ((1 + x_next) .* x_next - 14) - 29;
    
    % 1. F è la somma dei quadrati di tutto
    % MODIFICA 3: Sommiamo lungo la dimensione 1 (le righe)
    % F sarà un vettore riga 1xP con il valore della funzione per ogni punto
    F = 0.5 * (sum(f_odd.^2, 1) + sum(f_even.^2, 1));

    if nargout > 1

        % 2. Calcolo del Gradiente g = J' * f
        % Derivate prime dei residui rispetto a x_{i+1}
        % d_odd:  d/dx_next (xi + 5x_next^2 - x_next^3 - 2x_next - 13)
        d_odd  = 10*x_next - 3*x_next.^2 - 2;
        % d_even: d/dx_next (xi + x_next^2 + x_next^3 - 14x_next - 29)
        d_even = 3*x_next.^2 + 2*x_next - 14;
        
        % Costruzione del Giacobiano J (m x n) con m = 2(n-1)
        m = 2*(n-1);
        r = (1:m)';
        c1 = ceil(r/2);   % Indice per x_i
        c2 = c1 + 1;      % Indice per x_{i+1}
        
        v1 = ones(m, 1);  % df/dxi è sempre 1
        v2 = zeros(m, 1); % df/dx_next (alternato tra d_odd e d_even)
        v2(1:2:end) = d_odd;
        v2(2:2:end) = d_even;
        
        J = sparse([r; r], [c1; c2], [v1; v2], m, n);
        
        % Vettore residui completo
        f_vec = zeros(m, 1);
        f_vec(1:2:end) = f_odd;
        f_vec(2:2:end) = f_even;
        
        g = J' * f_vec;
    end

    if nargout > 2

        % 3. Calcolo dell'Hessiana esatta
        % Derivate seconde rispetto a x_{i+1}
        d2_odd  = 10 - 6*x_next;
        d2_even = 6*x_next + 2;
        
        % Termine di curvatura (diagonale): sum f_k * nabla^2 f_k
        % Solo la variabile x(j) con j > 1 ha derivate seconde (come x_{i+1})
        curv = zeros(n, 1);
        curv(2:end) = f_odd .* d2_odd + f_even .* d2_even;
        
        H = J' * J + spdiags(curv, 0, n, n);
    end
end