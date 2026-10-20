function g_fd = fwfd_gradChained(x, k, mode)
    % Calcolo ottimizzato del Gradiente Finite Differences per Chained F-R
    % Sfrutta la struttura sparsa: ogni x_i appare solo nei termini k=i e k=i-1
    
    n = length(x);
    base_h = 10^(-k);
    
    if strcmp(mode, 'fixed')
        h = ones(n, 1) * base_h;
    else
        h = base_h * max(abs(x), 1e-8);
    end
    
    % Definizione funzione energia locale per il blocco i
    % E dipende da u (x_current) e v (x_next)
    calc_E = @(u, v) ...
        0.5 * ( ...
            (u + v .* ((5 - v) .* v - 2) - 13).^2 + ...    % f_odd^2
            (u + v .* ((1 + v) .* v - 14) - 29).^2 ...     % f_even^2
        );

    % --- 1. Calcolo Energia Base ---
    % Usiamo x(1:n-1) come u e x(2:n) come v
    E_base = calc_E(x(1:n-1), x(2:n));
    
    % --- 2. Calcolo Gradienti ---
    % Quando perturbiamo x_k, esso agisce in due modi:
    % A) Come "x_current" nel blocco k
    % B) Come "x_next" nel blocco k-1
    
    g_fd = zeros(n, 1);
    
    % A) Perturbazione su x_current (indici 1:n-1)
    % Calcoliamo E(u+h, v)
    h_curr = h(1:n-1);
    E_pert_curr = calc_E(x(1:n-1) + h_curr, x(2:n));
    
    diff_curr = (E_pert_curr - E_base) ./ h_curr;
    g_fd(1:n-1) = g_fd(1:n-1) + diff_curr;
    
    % B) Perturbazione su x_next (indici 2:n)
    % Calcoliamo E(u, v+h)
    % Nota: qui stiamo guardando il blocco i, ma perturbiamo v (che è x_{i+1})
    h_next = h(2:n);
    E_pert_next = calc_E(x(1:n-1), x(2:n) + h_next);
    
    diff_next = (E_pert_next - E_base) ./ h_next;
    g_fd(2:n) = g_fd(2:n) + diff_next;
    
end