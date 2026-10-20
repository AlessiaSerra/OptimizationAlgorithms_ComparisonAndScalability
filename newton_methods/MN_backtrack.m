function [xk, fk, k, gradfk_norm, xseq, btseq] = MN_backtrack(x0, f, gradf, Hessf,...
                                                             c1, rho, kmax, btmax, jmax, tolgrad)
%Initialization
n = length(x0);
xseq = zeros(n,kmax);
btseq = zeros(1, kmax);

xk = x0;
k = 0;
fk = f(xk);
gradfk = gradf(xk);
gradfk_norm = norm(gradfk);

% --- NUOVI PARAMETRI PER STAGNAZIONE ---
    tolX = 1e-12;       % Soglia minima spostamento x
    tolF = 1e-12;       % Soglia minima decremento f
    max_patience = 150;   % Quante iterazioni di stagnazione tollerare
    patience_cnt = 0;   % Contatore attuale
    patience_tot = 0;
    % ---------------------------------------

while k < kmax && gradfk_norm >= tolgrad 
    
    %Compute the descent direction

    beta = 1e-4;
    hessfk = Hessf(xk); 
    lambda_min = min(diag(hessfk));
    successo = false;

    if(lambda_min > 0 )
        tao = 0;
    else
        tao = beta - lambda_min;
    end
    
    for j = 0:jmax

        Bk = hessfk + tao * speye(n);
        % Tentativo di Cholesky (flag = 0 se successo, != 0 se fallisce)
        [L, flag] = chol(Bk, 'lower');
        
        if flag == 0
            successo = true;
            break; % Scomposizione riuscita! Usciamo dal ciclo.
        else
            tao = max(2*tao, beta);
        end
    end
    if ~successo
        error('Impossibile rendere la matrice definita positiva.');
    end
    % DIRECT SOLVER usando i fattori di Cholesky
    % Vogliamo risolvere: Bk * p = -g
    % Sappiamo che Bk = L * L'
    % Quindi: L * (L' * p) = -g
    
    % Passo A: Sostituzione in avanti (Forward Substitution)
    % Risolviamo L * y = -g per trovare y
    % In MATLAB, l'operatore \ riconosce che L è triangolare inf. e ottimizza.
    y = L \ (-gradfk);
    
    % Passo B: Sostituzione all'indietro (Backward Substitution)
    % Risolviamo L' * p = y per trovare p
    pk = L' \ y;
    

    % %Debug for the modified hessian and new direction computed
    % cos_angle = (pk' * (-gradfk)) / (norm(pk) * norm(gradfk));
    % if tao > 0
    %     % Stima basata sul rapporto diag_max / diag_min Attuale
    %     %min_diag_val = min(diag(hessfk)) + taho;
    %     %max_diag_val = max(diag(hessfk)) + taho;
    %     %cond_approx = max_diag_val / min_diag_val; 
    %     cond_approx = condest(Bk);
    %     type_step = 'MODIFIED';
    % else
    %     %cos_angle = nan; % Se tau=0 è Newton puro, non ci interessa questo confronto
    %     cond_approx = condest(hessfk);
    %     type_step = 'PURE NEWTON';
    % end




    %BACKTRACKING
    alpha = 1;
    %Compute the candidate new xk
    xnew = xk + alpha*pk;
    %Compute the value of f in the candidate new xk
    fnew = f(xnew);

    c1_gradfk_pk = c1 * gradfk' * pk;
    bt = 0;
    while bt < btmax && fnew > (fk + alpha * c1_gradfk_pk)
        alpha = rho * alpha;
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        bt = bt+1;
    end
    
    %CHECK ON ARMIJO IF REACHED THE LAST STEP
    if bt == btmax && fnew > (fk + alpha * c1_gradfk_pk)
        fprintf('Backtracking failed. Passo troppo piccolo.\n');
        break
    end

    % --- LOGICA DI DETECTION E CORREZIONE ---
    rel_step_x = norm(xnew - xk) / (1 + norm(xk));
    rel_diff_f = abs(fnew - fk) / (1 + abs(fk));
    
    if rel_step_x < tolX || rel_diff_f < tolF
        patience_cnt = patience_cnt + 1;
    % else
    %     patience_cnt = 0;
        % Opzionale: se le cose vanno bene, potresti voler diminuire tao per l'iterazione successiva
        % tao = tao / 2; 
    end
    
      if patience_cnt >= max_patience
           fprintf('Stagnazione rilevata all''iterazione %d.\n', k);
           break;
        end

    %Compute the new value for xk and fk
    xk = xnew;
    fk = fnew;

    %Compute the gradient of f in  xk
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    
    % %STAMPA DI DEBUG: Visualizza progresso ogni 5-10 iterazioni
    % if mod(k, 5) == 0  % Stampa ogni  5 iterazioni
    %     fprintf('Iter: %d | %s | F: %.4e | ||g||: %.4e | Tau: %.2e | Stepsize: %.4e | Cos(ang)=%.4f (1=SD)| Cond~%.1e\n', ...
    %             k, type_step, fk, gradfk_norm, tao, alpha, cos_angle, cond_approx);
    % end

    k = k+1;
    %Store the current 
    xseq(:, k) = xk;
   
    %Store bt iterations in btseq
    btseq(k) = bt;
end

xseq = xseq(:,1:k);        %%% va bene?
btseq = btseq(:,1:k);      %%%% va bene?
end
