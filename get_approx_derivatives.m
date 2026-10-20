% Parametri da iterare
k_vals = [4, 8, 12];
modes = {'fixed', 'adaptive'};

root = fileparts(mfilename('fullpath'));
data_dir = fullfile(root, 'data');

for k_val = k_vals
    for m = 1:length(modes)
        mode_val = modes{m};
        
        % Definizione degli handle "approx" che catturano i valori correnti di k e mode
        % Chained Freudenstein-Roth
        Hessf1_approx1 = @(x) get_hess_c_approx1(x, k_val, mode_val);
        gradf1_approx2 = @(x) get_grad_c_approx2(x, k_val, mode_val);
        Hessf1_approx2 = @(x) get_hess_c_approx2(x, k_val, mode_val);
        
        % Broyden Tridiagonal
        Hessf2_approx1 = @(x) get_hess_b_approx1(x, k_val, mode_val);
        gradf2_approx2 = @(x) get_grad_b_approx2(x, k_val, mode_val);
        Hessf2_approx2 = @(x) get_hess_b_approx2(x, k_val, mode_val);
        
        % Generazione nome file dinamico (es. dati_k4_fixed.mat)
        filename = fullfile(data_dir, ...
            sprintf('dati_approx_k%d_%s.mat', k_val, mode_val));        
        % Salvataggio degli handle approssimati
        save(filename, 'Hessf1_approx1', 'gradf1_approx2', 'Hessf1_approx2', ...
                       'Hessf2_approx1', 'gradf2_approx2', 'Hessf2_approx2');
        
        fprintf('Creato file: %s\n', filename);
    end
end

%% --- FUNZIONI DI SUPPORTO (Corrette per richiamare i file FD) ---

% CHAINED FREUDENSTEIN-ROTH
function h = get_hess_c_approx1(x, k, mode)
    % Richiama l'Hessiana calcolata con gradiente esatto
    h = chained_derivatives_grad(x, k, mode);
end

function g = get_grad_c_approx2(x, k, mode)
    % Richiama il gradiente calcolato senza gradiente esatto (1° output)
    g = fwfd_gradChained(x, k, mode);
end

function h = get_hess_c_approx2(x, k, mode)
    % Richiama l'Hessiana calcolata senza gradiente esatto (2° output)
    h = fd_HessChained(x, k, mode);
end

% BROYDEN TRIDIAGONAL
function h = get_hess_b_approx1(x, k, mode)
    % Richiama l'Hessiana calcolata con gradiente esatto (2° output)
    [~, h] = broyden_derivatives_grad(x, k, mode);
end

function g = get_grad_b_approx2(x, k, mode)
    % Richiama il gradiente calcolato senza gradiente esatto (1° output)
    g = fwfd_gradBroyden(x, k, mode);
end

function h = get_hess_b_approx2(x, k, mode)
    % Richiama l'Hessiana calcolata senza gradiente esatto (2° output)
    h = fd_HessBroyden(x, k, mode);
end