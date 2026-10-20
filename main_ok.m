clear;
clc;

root = fileparts(mfilename('fullpath'));

addpath(genpath(fullfile(root, 'newton_methods')));
addpath(genpath(fullfile(root, 'derivatives')));
addpath(genpath(fullfile(root, 'functions')));
addpath(genpath(fullfile(root, 'data')));

% All'inizio dello script
filename = "risultati.txt";
if exist(filename, 'file'); delete(filename); end
diary(filename)
students_id = [359693; 362974];
rng(min(students_id));

n = 1e5; %[2, 1e3, 1e4, 1e5];
num_random_points = 5;
k_valss = [4,8,12];
modes = {'fixed','adaptive'};

get_approx_derivatives;

times = struct();


for i=1:length(n)

    %Generazione punti per Chained
    x0_chained_part = 0.5*ones(n(i)-1,1);
    x0_chained_sugg = [x0_chained_part; -2];

    start_points_chained = zeros(n(i), num_random_points+1);
    start_points_chained(:, 1) = x0_chained_sugg;    

    for j = 1:num_random_points
        % Generazione uniforme in [x_bar - 1, x_bar + 1]
        % Perturbazione in [-1, 1] = 2*rand - 1
        perturbation = 2 * rand(n(i), 1) - 1; 
        start_points_chained(:, j+1) = x0_chained_sugg + perturbation;
    end

    %Generazione punti per Broyden
    x0_broyden_sugg = -ones(n(i),1);
    start_points_broyden = zeros(n(i), num_random_points + 1);
    start_points_broyden(:, 1) = x0_broyden_sugg;
    
    for j = 1:num_random_points
        perturbation = 2 * rand(n(i), 1) - 1; 
        start_points_broyden(:, j+1) = x0_broyden_sugg + perturbation;
    end
        
    
    % INIZIALIZZO I PARAMETRI DA PASSARE ALLE FUNZIONI
    c1 = 1e-4;
    rho_Cmn = 0.5;
    rho_Ctn = 0.5;
    rho_Bmn = 0.1;
    rho_Btn = 0.5;
    kmax = 1500;
    btmax = 50;
    jmax = 80;
    fprintf("TUNING CON rho_Cmn: %.2f", rho_Cmn);

    %Scalatura della tolleranza in base alla dimensione
    if n(i) == 2
        tolgrad = 1e-8;
    else
        raw_tol = 1e-8 * sqrt(n(i));
        tolgrad = round(raw_tol, 1, 'significant');
    end
    fprintf("\n=== ANALISI PER n = %d | tolgrad = %2.e===\n", n(i), tolgrad)% aggiunta enri


    get_functions;

    for ik = 1:length(k_valss)
        k = k_valss(ik);

        % associa l'indice ik al valore reale di k
        times(ik).k = k;

        %Scalatura della tolleranza in base a k
        if k==4
            base_tol = 1e-4;
        elseif k==8
            base_tol = 1e-6; 
        elseif k==12
            base_tol = 1e-8;
        end

        %Scalatura della tolleranza in base alla dimensione n
        row_tol_k = base_tol * sqrt(n(i));

        tolgrad_k = round(row_tol_k,1, 'significant');

        for im = 1:length(modes)
            mode = modes{im};
    
            fprintf('\n==============================\n');
            fprintf('k = %d | mode = %s\n', k, mode);
            fprintf('==============================\n');
    
            % carico le approssimazioni giuste
            filename = fullfile(root, 'data', ...
            sprintf('dati_approx_k%d_%s.mat', k, mode));
            load(filename);    
            
            %per avere i function handle
        

            %Inizializzazione variabili per contare i successi:
            %Per Chained modified newton
            successi_Cmn = 0;
            iter_Cmn =0;
            successi_Cmn_a1 = 0;
            iter_Cmn_a1 = 0;
            successi_Cmn_a2 = 0;
            iter_Cmn_a2 = 0;
            %Per Chained truncated newton
            successi_Ctn = 0;
            iter_Ctn = 0;
            successi_Ctn_a1 = 0;
            iter_Ctn_a1=0;
            successi_Ctn_a2 = 0;
            iter_Ctn_a2=0;
            %Per Broyden modified newton
            successi_Bmn = 0;
            iter_Bmn =0;
            successi_Bmn_a1 = 0;
            iter_Bmn_a1 = 0;
            successi_Bmn_a2 = 0;
            iter_Bmn_a2 = 0;
            %Per Chained truncated newton
            successi_Btn = 0;
            iter_Btn = 0;
            successi_Btn_a1 = 0;
            iter_Btn_a1=0;
            successi_Btn_a2 = 0;
            iter_Btn_a2=0;

        
        fprintf("\nCalcolo Chained Freudstein...\n")
        %FOR PER VALUTARE I METODI DI OTTIMIZZAZIONE
        for j = 1:size(start_points_chained, 2)
                fprintf("STARTING POINT %d: [%.2f; %.2f]\n",j, start_points_chained(1, j), start_points_chained(2, j));% modifica enri
                fprintf("--Modified newton--\n")
                x0 = start_points_chained(:, j);

                %modified
                t_start = tic;
                [xk_Cmn, fk_Cmn, k_Cmn, gradfk_norm_Cmn, xseq_Cmn, btseq_Cmn] = MN_backtrack(x0, f1, gradf1, Hessf1, c1, rho_Cmn, kmax, btmax, jmax, tolgrad);
                times(ik).(mode).ch.MN.exact(i,j) = toc(t_start);
                x_best_ch = xseq_Cmn(:, k_Cmn);
                p_exp = compute_roc(xseq_Cmn, x_best_ch);
%                 fk_Cmn
%                 gradfk_norm_Cmn
%                 k_Cmn
                if gradfk_norm_Cmn < tolgrad
                    fprintf('|MN Chained Exact derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: %.4f | time: %.4f | CONVERGENCE\n', fk_Cmn, gradfk_norm_Cmn, k_Cmn, tolgrad, p_exp, times(ik).(mode).ch.MN.exact(i,j));
                    successi_Cmn = successi_Cmn +1;
                    iter_Cmn = iter_Cmn + k_Cmn;
                else
                    fprintf('|MN Chained Exact derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: N/A | time: %.4f| NO  CONVERGENCE \n', fk_Cmn, gradfk_norm_Cmn, k_Cmn, tolgrad, times(ik).(mode).ch.MN.exact(i,j));
                end

                %modified with derivatives approximated using exact gradient
                t_start = tic;
                [xk_Cmn_a1, fk_Cmn_a1, k_Cmn_a1, gradfk_norm_Cmn_a1, xseq_Cmn_a1, btseq_Cmn_a1] = MN_backtrack(x0, f1, gradf1, Hessf1_approx1, c1, rho_Cmn, kmax, btmax, jmax, tolgrad_k);
                times(ik).(mode).ch.MN.approx1(i,j) = toc(t_start);
                p_exp = compute_roc(xseq_Cmn_a1, x_best_ch);

%                 fk_Cmn_a1
%                 gradfk_norm_Cmn_a1
%                 k_Cmn_a1
                if gradfk_norm_Cmn_a1 < tolgrad_k
                    fprintf('|MN Chained Approx Hess, exact. Grad | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: %.4f  | time: %.4f | CONVERGENCE\n', fk_Cmn_a1, gradfk_norm_Cmn_a1, k_Cmn_a1, tolgrad_k, p_exp, times(ik).(mode).ch.MN.approx1(i,j));
                    successi_Cmn_a1 = successi_Cmn_a1 +1;
                    iter_Cmn_a1 = iter_Cmn_a1 + k_Cmn_a1;
                else
                    fprintf('|MN Chained Approx Hess, exact. Grad | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: N/A | time: %.4f | NO CONVERGENCE\n', fk_Cmn_a1, gradfk_norm_Cmn_a1, k_Cmn_a1, tolgrad_k, times(ik).(mode).ch.MN.approx1(i,j));
                end
                %
                %modified with all derivatives approimated
                t_start = tic;
                [xk_Cmn_a2, fk_Cmn_a2, k_Cmn_a2, gradfk_norm_Cmn_a2, xseq_Cmn_a2, btseq_Cmn_a2] = MN_backtrack(x0, f1, gradf1_approx2, Hessf1_approx2, c1, rho_Cmn, kmax, btmax, jmax, tolgrad_k);
                times(ik).(mode).ch.MN.approx2(i,j) = toc(t_start);
                p_exp = compute_roc(xseq_Cmn_a2, x_best_ch);

%                 fk_Cmn_a2
%                 gradfk_norm_Cmn_a2
%                 k_Cmn_a2
                if gradfk_norm_Cmn_a2 < tolgrad_k
                    fprintf('|MN Chained Approx all derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: %.4f  | time: %.4f | CONVERGENCE\n', fk_Cmn_a2, gradfk_norm_Cmn_a2, k_Cmn_a2, tolgrad_k, p_exp, times(ik).(mode).ch.MN.approx2(i,j));
                    successi_Cmn_a2 = successi_Cmn_a2 +1;
                    iter_Cmn_a2 = iter_Cmn_a2 + k_Cmn_a2;
                else
                    fprintf('|MN Chained Approx all derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: N/A | time: %.4f | NO CONVERGENCE\n', fk_Cmn_a2, gradfk_norm_Cmn_a2, k_Cmn_a2, tolgrad_k, times(ik).(mode).ch.MN.approx2(i,j));
                end 
                %
                fprintf("--Truncated newton--\n")
                t_start = tic;

                %truncated with exact derivatives
                [xk_Ctn, fk_Ctn, k_Ctn, gradfk_norm_Ctn, xseq_Ctn, btseq_Ctn] = truncated_newton(x0, f1, gradf1, Hessf1, c1, rho_Ctn, kmax, btmax, tolgrad);      
                times(ik).(mode).ch.TN.exact(i,j) = toc(t_start);
                p_exp = compute_roc(xseq_Ctn, x_best_ch);
                
%                 fk_Ctn
%                 gradfk_norm_Ctn
%                 k_Ctn
                if gradfk_norm_Ctn < tolgrad
                    fprintf('|TN Chained Exact derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: %.4f  | time: %.4f | CONVERGENCE \n', fk_Ctn, gradfk_norm_Ctn, k_Ctn, tolgrad, p_exp, times(ik).(mode).ch.TN.exact(i,j));
                    successi_Ctn = successi_Ctn +1;
                    iter_Ctn = iter_Ctn + k_Ctn;
                else
                    fprintf('|TN Chained Exact derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: N/A | time: %.4f |NO CONVERGENCE \n', fk_Ctn, gradfk_norm_Ctn, k_Ctn, tolgrad, times(ik).(mode).ch.TN.exact(i,j));
                end


                %truncated with derivatives approximated using exact gradient
                t_start = tic;
                [xk_Ctn_a1, fk_Ctn_a1, k_Ctn_a1, gradfk_norm_Ctn_a1, xseq_Ctn_a1, btseq_Ctn_a1] = truncated_newton(x0, f1, gradf1, Hessf1_approx1, c1, rho_Ctn, kmax, btmax, tolgrad_k);
                times(ik).(mode).ch.TN.approx1(i,j) = toc(t_start);
                p_exp = compute_roc(xseq_Ctn_a1, x_best_ch);
%                 fk_Ctn_a1
%                 gradfk_norm_Ctn_a1
%                 k_Ctn_a1
                if gradfk_norm_Ctn_a1 < tolgrad_k
                    fprintf('|TN Chained Approx Hess, exact. Grad | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: %.4f  | time: %.4f | CONVERGENCE\n', fk_Ctn_a1, gradfk_norm_Ctn_a1, k_Ctn_a1, tolgrad_k, p_exp, times(ik).(mode).ch.TN.approx1(i,j));
                    successi_Ctn_a1 = successi_Ctn_a1 +1;
                    iter_Ctn_a1 = iter_Ctn_a1 + k_Ctn_a1;
                else
                    fprintf('|TN Chained Approx Hess, exact. Grad | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: N/A | time: %.4f | NO CONVERGENCE\n', fk_Ctn_a1, gradfk_norm_Ctn_a1, k_Ctn_a1, tolgrad_k, times(ik).(mode).ch.TN.approx1(i,j));
                end   
                %
                %truncated with all derivatives approimated
                t_start = tic;
                [xk_Ctn_a2, fk_Ctn_a2, k_Ctn_a2, gradfk_norm_Ctn_a2, xseq_Ctn_a2, btseq_Ctn_a2] = truncated_newton(x0, f1, gradf1_approx2, Hessf1_approx2, c1, rho_Ctn, kmax, btmax, tolgrad_k);
                times(ik).(mode).ch.TN.approx2(i,j) = toc(t_start);
                p_exp = compute_roc(xseq_Ctn_a2, x_best_ch);
%                 fk_Ctn_a2
%                 gradfk_norm_Ctn_a2
%                 k_Ctn_a2
                if gradfk_norm_Ctn_a2 < tolgrad_k
                    fprintf('|TN Chained Approx all derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: %.4f  | time: %.4f | CONVERGENCE\n', fk_Ctn_a2, gradfk_norm_Ctn_a2, k_Ctn_a2, tolgrad_k, p_exp, times(ik).(mode).ch.TN.approx2(i,j));
                    successi_Ctn_a2 = successi_Ctn_a2 +1;
                    iter_Ctn_a2 = iter_Ctn_a2 + k_Ctn_a2;
                else
                    fprintf('|TN Chained Approx all derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: N/A | time: %.4f | NO CONVERGENCE\n', fk_Ctn_a2, gradfk_norm_Ctn_a2, k_Ctn_a2, tolgrad_k, times(ik).(mode).ch.TN.approx2(i,j));
                end
                %
    
        end
        
        %Stampa statistiche a video
        if successi_Cmn > 0
            iter_Cmn = iter_Cmn/successi_Cmn;
            fprintf('\nMN Chained Exact derivatives  | Punti convergenti: %d/%d | Iterazioni medie (sui successi): %.1f\n', ...
                successi_Cmn, size(start_points_chained, 2), iter_Cmn);
        else
            fprintf('\nMN Chained Exact derivatives  | Punti convergenti: 0/%d\n', ...
                size(start_points_chained, 2));
        end
        if successi_Cmn_a1 >0
            iter_Cmn_a1 = iter_Cmn_a1/successi_Cmn_a1;
            fprintf('\nMN Chained Approx Hess, exact. Grad | Punti convergenti: %d/%d | Iterazioni medie (sui successi): %.1f\n', ...
                successi_Cmn_a1, size(start_points_chained, 2), iter_Cmn_a1);
        else
            fprintf('\nMN Chained Approx Hess, exact. Grad | Punti convergenti: 0/%d\n', ...
                size(start_points_chained, 2));
        end
        if successi_Cmn_a2 >0
            iter_Cmn_a2 = iter_Cmn_a2/successi_Cmn_a2;
            fprintf('\nMN Chained Approx all derivatives | Punti convergenti: %d/%d | Iterazioni medie (sui successi): %.1f\n', ...
                successi_Cmn_a2, size(start_points_chained, 2), iter_Cmn_a2);
        else
            fprintf('\nMN Chained Approx all derivatives | Punti convergenti: 0/%d\n', ...
                size(start_points_chained, 2));
        end

        %truncated
        if successi_Ctn > 0
            iter_Ctn = iter_Ctn/successi_Ctn;
            fprintf('\nTN Chained Exact derivatives  | Punti convergenti: %d/%d | Iterazioni medie (sui successi): %.1f\n', ...
                successi_Ctn, size(start_points_chained, 2), iter_Ctn);
        else
            fprintf('\nTN Chained Exact derivatives  | Punti convergenti: 0/%d\n', ...
                size(start_points_chained, 2));
        end
        if successi_Ctn_a1 >0
            iter_Ctn_a1 = iter_Ctn_a1/successi_Ctn_a1;
            fprintf('\nTN Chained Approx. Hess, exact. Grad | Punti convergenti: %d/%d | Iterazioni medie (sui successi): %.1f\n', ...
                successi_Ctn_a1, size(start_points_chained, 2), iter_Ctn_a1);
        else
            fprintf('\nTN Chained Approx. Hess, exact. Grad | Punti convergenti: 0/%d\n', ...
                size(start_points_chained, 2));
        end
        if successi_Ctn_a2 >0
            iter_Ctn_a2 = iter_Ctn_a2/successi_Ctn_a2;
            fprintf('\nTN Chained Approx all derivatives | Punti convergenti: %d/%d | Iterazioni medie (sui successi): %.1f\n', ...
                successi_Ctn_a2, size(start_points_chained, 2), iter_Ctn_a2);
        else
            fprintf('\nTN Chained Approx all derivatives | Punti convergenti: 0/%d\n', ...
                size(start_points_chained, 2));
        end
        

       % FACCIO LO STESSO PER BROYDEN
   

        fprintf("\n************** PER BROYDEN **************\n")
        for j = 1:size(start_points_broyden, 2)
                x0 = start_points_broyden(:, j);
                fprintf("STARTING POINT %d: [%.2f; %.2f]\n",j, start_points_broyden(1, j), start_points_broyden(2, j));
                fprintf("--Modified newton--\n") 

                %modified with exact derivatives
                t_start = tic;
                [xk_Bmn, fk_Bmn, k_Bmn, gradfk_norm_Bmn, xseq_Bmn, btseq_Bmn] = MN_backtrack(x0, f2, gradf2, Hessf2, c1, rho_Bmn, kmax, btmax, jmax, tolgrad);
                times(ik).(mode).br.MN.exact(i,j) = toc(t_start);
                x_best_br = xseq_Bmn(:, k_Bmn);
                p_exp = compute_roc(xseq_Bmn, x_best_br);
                %fk_Bmn
                %gradfk_norm_Bmn
                %k_Bmn
                if gradfk_norm_Bmn < tolgrad
                    fprintf('|MN Broyden Exact derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: %.4f  | time: %.4f | CONVERGENCE\n', fk_Bmn, gradfk_norm_Bmn, k_Bmn, tolgrad, p_exp, times(ik).(mode).br.MN.exact(i,j));
                    successi_Bmn = successi_Bmn +1;
                    iter_Bmn = iter_Bmn + k_Bmn;
                else
                    fprintf('|MN Broyden Exact derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: N/A | time: %.4f | NO CONVERGENCE \n', fk_Bmn, gradfk_norm_Bmn, k_Bmn, tolgrad, times(ik).(mode).br.MN.exact(i,j));

                end

                %modified with derivatives approximated using exact gradient
                t_start = tic;
                [xk_Bmn_a1, fk_Bmn_a1, k_Bmn_a1, gradfk_norm_Bmn_a1, xseq_Bmn_a1, btseq_Bmn_a1] = MN_backtrack(x0, f2, gradf2, Hessf2_approx1, c1, rho_Bmn, kmax, btmax, jmax, tolgrad_k);
                times(ik).(mode).br.MN.approx1(i,j) = toc(t_start);
                p_exp = compute_roc(xseq_Bmn_a1, x_best_br);
                %fk_Bmn_a1
                %gradfk_norm_Bmn_a1
                %k_Bmn_a1
                if gradfk_norm_Bmn_a1 < tolgrad_k
                    fprintf('|MN Broyden Approx Hess, exact. Grad | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: %.4f  | time: %.4f | CONVERGENCE\n', fk_Bmn_a1, gradfk_norm_Bmn_a1, k_Bmn_a1, tolgrad_k, p_exp, times(ik).(mode).br.MN.approx1(i,j));
                    successi_Bmn_a1 = successi_Bmn_a1 +1;
                    iter_Bmn_a1 = iter_Bmn_a1 +k_Bmn_a1;
                else
                    fprintf('|MN Broyden Approx Hess, exact. Grad | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: N/A | time: %.4f | NO CONVERGENCE\n', fk_Bmn_a1, gradfk_norm_Bmn_a1, k_Bmn_a1, tolgrad_k, times(ik).(mode).br.MN.approx1(i,j));
                end

                %
                %modified with all derivatives approimated
                t_start = tic;
                [xk_Bmn_a2, fk_Bmn_a2, k_Bmn_a2, gradfk_norm_Bmn_a2, xseq_Bmn_a2, btseq_Bmn_a2] = MN_backtrack(x0, f2, gradf2_approx2, Hessf2_approx2, c1, rho_Bmn, kmax, btmax, jmax, tolgrad_k);
                times(ik).(mode).br.MN.approx2(i,j) = toc(t_start);
                p_exp = compute_roc(xseq_Bmn_a2, x_best_br);
                %fk_Bmn_a2
                %gradfk_norm_Bmn_a2
                %k_Bmn_a2
                if gradfk_norm_Bmn_a2 < tolgrad_k
                    fprintf('|MN Broyden Approx all derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: %.4f  | time: %.4f | CONVERGENCE\n', fk_Bmn_a2, gradfk_norm_Bmn_a2, k_Bmn_a2, tolgrad_k, p_exp, times(ik).(mode).br.MN.approx2(i,j));
                    successi_Bmn_a2 = successi_Bmn_a2 +1;
                    iter_Bmn_a2 = iter_Bmn_a2 + k_Bmn_a2;
                else
                    fprintf('|MN Broyden Approx all derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: N/A | time: %.4f | NO CONVERGENCE\n', fk_Bmn_a2, gradfk_norm_Bmn_a2, k_Bmn_a2, tolgrad_k, times(ik).(mode).br.MN.approx2(i,j));
                end

                fprintf("--Truncated newton--\n")
                %truncated
                t_start = tic;
                [xk_Btn, fk_Btn, k_Btn, gradfk_norm_Btn, xseq_Btn, btseq_Btn] = truncated_newton(x0, f2, gradf2, Hessf2, c1, rho_Btn, kmax, btmax, tolgrad);
                times(ik).(mode).br.TN.exact(i,j) = toc(t_start);
                p_exp = compute_roc(xseq_Btn, x_best_br);
                %fk_Btn
                %gradfk_norm_Btn
                %k_Btn
                if gradfk_norm_Btn < tolgrad
                    fprintf('|TN Broyden Exact derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: %.4f  | time: %.4f | CONVERGENCE \n', fk_Btn, gradfk_norm_Btn, k_Btn, tolgrad, p_exp, times(ik).(mode).br.TN.exact(i,j));
                    successi_Btn = successi_Btn +1;
                    iter_Btn = iter_Btn + k_Btn;
                else
                    fprintf('|TN Broyden Exact derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: N/A | time: %.4f | NO CONVERGENCE \n', fk_Btn, gradfk_norm_Btn, k_Btn, tolgrad, times(ik).(mode).br.TN.exact(i,j));
                end


                %truncated with derivatives approximated using exact gradient
                t_start = tic;
                [xk_Btn_a1, fk_Btn_a1, k_Btn_a1, gradfk_norm_Btn_a1, xseq_Btn_a1, btseq_Btn_a1] = truncated_newton(x0, f2, gradf2, Hessf2_approx1, c1, rho_Btn, kmax, btmax, tolgrad_k);
                times(ik).(mode).br.TN.approx1(i,j) = toc(t_start);
                p_exp = compute_roc(xseq_Btn_a1, x_best_br);
                %fk_Btn_a1
                %gradfk_norm_Btn_a1
                %k_Btn_a1
                if gradfk_norm_Btn_a1 < tolgrad_k
                    fprintf('|TN Broyden Approx Hess, exact. Grad | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: %.4f | time: %.4f | CONVERGENCE\n', fk_Btn_a1, gradfk_norm_Btn_a1, k_Btn_a1, tolgrad_k, p_exp, times(ik).(mode).br.TN.approx1(i,j));
                    successi_Btn_a1 = successi_Btn_a1 +1;
                    iter_Btn_a1 = iter_Btn_a1 + k_Btn_a1;
                else
                    fprintf('|TN Broyden Approx Hess, exact. Grad | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: N/A | time: %.4f | NO CONVERGENCE\n', fk_Btn_a1, gradfk_norm_Btn_a1, k_Btn_a1, tolgrad_k, times(ik).(mode).br.TN.approx1(i,j));
                end

                %
                %truncated with all derivatives approimated
                t_start = tic;
                [xk_Btn_a2, fk_Btn_a2, k_Btn_a2, gradfk_norm_Btn_a2, xseq_Btn_a2, btseq_Btn_a2] = truncated_newton(x0, f2, gradf2_approx2, Hessf2_approx2, c1, rho_Btn, kmax, btmax, tolgrad_k);
                times(ik).(mode).br.TN.approx2(i,j) = toc(t_start);
                p_exp = compute_roc(xseq_Btn_a2, x_best_br);
                %fk_Btn_a2
                %gradfk_norm_Btn_a2
                %k_Btn_a2
                if gradfk_norm_Btn_a2 < tolgrad_k
                    fprintf('|TN Broyden Approx all derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: %.4f  | time: %.4f | CONVERGENCE\n', fk_Btn_a2, gradfk_norm_Btn_a2, k_Btn_a2, tolgrad_k, p_exp, times(ik).(mode).br.TN.approx2(i,j));
                    successi_Btn_a2 = successi_Btn_a2 +1;
                    iter_Btn_a2 = iter_Btn_a2 + k_Btn_a2;
                else
                    fprintf('|TN Broyden Approx all derivatives | F: %.4e | ||g||: %.4e | Iters: %d | tolgrad = %2.e | ROC: N/A | time: %.4f | NO CONVERGENCE\n', fk_Btn_a2, gradfk_norm_Btn_a2, k_Btn_a2, tolgrad_k, times(ik).(mode).br.TN.approx2(i,j));
                end
                %
    
        end
      % Stampa statistiche a video per broyden
%        modified
        if successi_Bmn > 0
            iter_Bmn = iter_Bmn/successi_Bmn;
            fprintf('\nMN Broyden Exact derivatives  | Punti convergenti: %d/%d | Iterazioni medie (sui successi): %.1f\n', ...
                    successi_Bmn, size(start_points_broyden, 2), iter_Bmn);
        else
            fprintf('\nMN Broyden Exact derivatives  | Punti convergenti: 0/%d\n', ...
                    size(start_points_broyden, 2));
        end
        if successi_Bmn_a1 >0
            iter_Bmn_a1 = iter_Bmn_a1/successi_Bmn_a1;
            fprintf('\nMN Broyden Approx Hess, exact. Grad | Punti convergenti: %d/%d | Iterazioni medie (sui successi): %.1f\n', ...
                    successi_Bmn_a1, size(start_points_broyden, 2), iter_Bmn_a1);
        else
            fprintf('\nMN Broyden Approx Hess, exact. Grad | Punti convergenti: 0/%d\n', ...
                    size(start_points_broyden, 2));
        end
        if successi_Bmn_a2 >0
            iter_Bmn_a2 = iter_Bmn_a2/successi_Bmn_a2;
            fprintf('\nMN Broyden Approx all derivatives | Punti convergenti: %d/%d | Iterazioni medie (sui successi): %.1f\n', ...
                    successi_Bmn_a2, size(start_points_broyden, 2), iter_Bmn_a2);
        else
            fprintf('\nMN Broyden Approx all derivatives | Punti convergenti: 0/%d\n', ...
                    size(start_points_broyden, 2));
        end
        %truncated
        if successi_Btn > 0
            iter_Btn = iter_Btn/successi_Btn;
            fprintf('\nTN Bryden Exact derivatives  | Punti convergenti: %d/%d | Iterazioni medie (sui successi): %.1f\n', ...
                successi_Btn, size(start_points_broyden, 2), iter_Btn);
        else
            fprintf('\nTN Bryden Exact derivatives  | Punti convergenti: 0/%d\n', ...
                size(start_points_broyden, 2));
        end
        if successi_Btn_a1 >0
            iter_Btn_a1 = iter_Btn_a1/successi_Btn_a1;
            fprintf('\nTN Bryden Approx. Hess, exact. Grad | Punti convergenti: %d/%d | Iterazioni medie (sui successi): %.1f\n', ...
                successi_Btn_a1, size(start_points_broyden, 2), iter_Btn_a1);
        else
            fprintf('\nTN Chained Approx. Hess, exact. Grad | Punti convergenti: 0/%d\n', ...
                size(start_points_broyden, 2));
        end
        if successi_Btn_a2 >0
            iter_Btn_a2 = iter_Btn_a2/successi_Btn_a2;
            fprintf('\nTN Bryden Approx all derivatives | Punti convergenti: %d/%d | Iterazioni medie (sui successi): %.1f\n', ...
                successi_Btn_a2, size(start_points_broyden, 2), iter_Btn_a2);
        else
            fprintf('\nTN Bryden Approx all derivatives | Punti convergenti: 0/%d\n', ...
                size(start_points_broyden, 2));
        end


        end
     end

end

script_tempi;

diary off

