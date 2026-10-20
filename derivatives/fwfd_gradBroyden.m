function g_fd = fwfd_gradBroyden(x, k, mode)
    % x: punto in cui calcolare le derivate
    % k: esponente per l'incremento (4, 8, 12)
    % mode: 'fixed' per h costante, 'variable' per h_i proporzionale a x_i
    
    n = length(x);
    base_h = 10^(-k);
    
    % 1. Definizione dei vettori di incremento h
    if strcmp(mode, 'fixed')
        h = ones(n, 1) * base_h;
    else
        h = base_h * max(abs(x), 1e-8);
    end

    % Definizione della funzione locale f_k (componente interna)
    % f_val = (3 - 2*xc).*xc - xp - 2*xn + 1;
    calc_fk = @(xp, xc, xn) (3 - 2*xc).*xc - xp - 2*xn + 1;

    % Creiamo vettori estesi con zero padding per gestire i bordi (i-1 e i+1)
    % X_pad = [0, x_1, ..., x_n, 0]
    X = [0; x; 0];
    
    % Calcoliamo i termini f_k di base (non perturbati)
    % Indici in X: xp=X(1:n), xc=X(2:n+1), xn=X(3:n+2)
    f_base = calc_fk(X(1:n), X(2:n+1), X(3:n+2));
    E_base = 0.5 * f_base.^2;
    
    % --- Calcolo Vettoriale delle Differenze ---
    % Quando perturbiamo x_i (che corrisponde a X(i+1)), influenziamo:
    % 1. f_{i-1} (dove x_i è 'xn', il termine next)
    % 2. f_{i}   (dove x_i è 'xc', il termine current)
    % 3. f_{i+1} (dove x_i è 'xp', il termine prev)

    % Vettori di perturbazione h estesi
    H = [0; h; 0];

    % Termine 1: f_{i-1} modificato da x_i (perturbazione su xn)
    % Indici originali k=1..n. Qui guardiamo f_{k-1}. 
    % Usiamo shift: f_new calcolato usando X(3:n+2)+h come 'xn'
    f_prev_h = calc_fk(X(1:n), X(2:n+1), X(3:n+2) + H(3:n+2)); 
    
    % Termine 2: f_{i} modificato da x_i (perturbazione su xc)
    f_curr_h = calc_fk(X(1:n), X(2:n+1) + H(2:n+1), X(3:n+2));
    
    % Termine 3: f_{i+1} modificato da x_i (perturbazione su xp)
    f_next_h = calc_fk(X(1:n) + H(1:n), X(2:n+1), X(3:n+2));
    
    % Calcolo delle energie (0.5 * f^2) perturbate
    E_prev_h = 0.5 * f_prev_h.^2;
    E_curr_h = 0.5 * f_curr_h.^2;
    E_next_h = 0.5 * f_next_h.^2;

    % Ora assembliamo il gradiente g(i).
    % g(i) è la somma delle variazioni dei 3 termini coinvolti.
    % Dobbiamo allineare gli indici correttamente.
    
    % Diff su f_{i-1} (che è f_base(i-1)):
    diff_prev = [0; E_prev_h(1:n-1) - E_base(1:n-1)]; 
    
    % Diff su f_{i} (che è f_base(i)):
    diff_curr = E_curr_h - E_base;
    
    % Diff su f_{i+1} (che è f_base(i+1)):
    diff_next = [E_next_h(2:n) - E_base(2:n); 0];
    
    % Gradiente totale
    g_fd = (diff_prev + diff_curr + diff_next) ./ h;
  
end


  %{
    % Valore della funzione nel punto base
    F0 = broyden_tridiagonal(x);
    
    g_fd = zeros(n, 1);
    %Fi_plus_grad = zeros(n, 1); 
    %Fi_minus_grad = zeros(n, 1); 
    
    % 2. Calcolo del Gradiente (Forward Differenc) - Passo h
    for i = 1:n
        xh_plus = x;
        xh_plus(i) = xh_plus(i) + h(i);
        Fi_plus = broyden_tridiagonal(xh_plus);
        g_fd(i) = (Fi_plus - F0) / h(i);
    end
    fprintf('Fine calcolo Gradiente\n')
    %}
    
    %{
    % 2. Calcolo del Gradiente (CENTRAL Differenc) - Passo h
    % Richiede 2*n valutazioni di funzione
    for i = 1:n
        % Valutazione in x + h
        xh_plus = x;
        xh_plus(i) = xh_plus(i) + h(i);
        Fi_plus_grad = broyden_tridiagonal(xh_plus);
        
        % Valutazione in x - h
        xh_minus = x;
        xh_minus(i) = xh_minus(i) - h(i);
        Fi_minus_grad = broyden_tridiagonal(xh_minus);
        
        % Formula Differenza Centrale: (F(x+h) - F(x-h)) / (2*h)
        g_fd(i) = (Fi_plus_grad - Fi_minus_grad) / (2 * h(i));
    end
    %}