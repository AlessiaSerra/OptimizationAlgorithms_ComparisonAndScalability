function [xk, fk, k, gradfk_norm, xseq, btseq] = truncated_newton(x0, f, gradf, Hessf, ...
                                                             c1, rho, kmax, btmax, tolgrad)
    % --- NUOVI PARAMETRI PER STAGNAZIONE ---
    tolX = 1e-12;       % Soglia minima spostamento x
    tolF = 1e-12;       % Soglia minima decremento f
    max_patience = 150;   % Quante iterazioni di stagnazione tollerare
    patience_cnt = 0;   % Contatore attuale
    
    % Inizializzazione
    xseq = zeros(length(x0), kmax);
    btseq = zeros(1, kmax);
    
    xk = x0;
    fk = f(xk);
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    
    k = 0; % Contatore iterazioni (parte da 0 per coerenza logica)
      
    % Loop esterno (Newton)
    while k < kmax && gradfk_norm >= tolgrad
        % --- INIZIO CONJUGATE GRADIENT (Inner Loop) ---
        Bk = Hessf(xk);
        
        % Inizializzazione CG
        % Vogliamo risolvere: Bk * p = -gradfk
        z = zeros(length(xk), 1);   % z è la direzione accumulata
        r = -gradfk;                % Residuo iniziale (r0 = b - A*x0 = -g - 0)
        d = r;                      % Prima direzione di ricerca del CG
        
        % I correlate the era_k to the current tollerance
        eta_k = max(min(0.5, sqrt(gradfk_norm)), tolgrad*0.1); % sqrt per convergenza superlineare
        %eta_k = (min(0.5, sqrt(gradfk_norm)));

        % Default: se il CG fallisce subito, usa il gradiente (Steepest Descent)
        pk = -gradfk; 
        
        
        for j = 1:length(xk)
            Bd = Bk * d;
            curvature = d' * Bd;
            
            % 1. Controllo Curvatura
            if curvature > 0
                % Passo CG standard
                alpha_cg = (r' * r) / curvature;
                z_new = z + alpha_cg * d;
                
                % Aggiornamento residuo
                r_new = r - alpha_cg * Bd; 
            else
                % Curvatura negativa
                if j == 1
                    pk = -gradfk; 
                else
                    pk = z; 
                end
                break; % Troncamento
            end
            
            % 2. Controllo Convergenza (Spostato PRIMA di calcolare beta e d)
            if norm(r_new) <= eta_k * gradfk_norm
                pk = z_new; 
                break; % Convergenza raggiunta, evitiamo calcoli inutili
            end
            
            % 3. Aggiornamento direzione (solo se non abbiamo finito)
            beta_cg = (r_new' * r_new) / (r' * r);
            d = r_new + beta_cg * d;
            
            % Aggiornamento variabili per la prossima iterazione
            z = z_new;
            r = r_new;
            
            % Se arriviamo all'ultima iterazione senza break
            if j == length(xk)
                pk = z;
            end
        end
        % --- FINE INNER LOOP ---
        
        % Step 2.3: Backtracking (Armijo Line Search)
        alpha = 1;
        xnew = xk + alpha * pk;
        fnew = f(xnew);

        c1_gradfk_pk = c1 * gradfk' * pk;
        
        bt = 0;
        while bt < btmax && fnew > (fk + alpha * c1_gradfk_pk)
            alpha = rho * alpha;
            xnew = xk + alpha * pk;
            fnew = f(xnew);
            bt = bt + 1;
        end
         
        if bt == btmax && fnew > (fk + alpha * c1_gradfk_pk)
            fprintf('Backtracking failed. Passo troppo piccolo.\n');
            break
        end
       
        % 3. LOGICA DI DETECTION STAGNAZIONE
        rel_step_x = norm(xnew - xk) / (1 + norm(xk));
        rel_diff_f = abs(fnew - fk) / (1 + abs(fk));

        if rel_step_x < tolX || rel_diff_f < tolF
            patience_cnt = patience_cnt + 1;
        % else
        %     patience_cnt = 0; 
        % end

        if patience_cnt >= max_patience
           fprintf('Stagnazione rilevata all''iterazione %d.\n', k);
           break;
        end
        % Aggiornamento variabili
        xk = xnew;
        fk = fnew;
        gradfk = gradf(xk);
        gradfk_norm = norm(gradfk);
        
         
        % STAMPA DI DEBUG: Visualizza progresso ogni 5-10 iterazioni
        % if mod(k, 5) == 0 || k == 1
        % fprintf('   Iter %d: f(x) = %.4e, |g| = %.4e, bt = %d, eta = %.4e\n', ...
        %         k, fk, gradfk_norm, bt, eta_k);
        % end
         

        k = k + 1;

        % Store history
        xseq(:, k) = xk;
        btseq(k) = bt;
    end
    
    % Trim output arrays
    xseq = xseq(:, 1:k);
    btseq = btseq(1:k);
end