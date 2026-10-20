root = fileparts(mfilename('fullpath'));
data_dir = fullfile(root, 'data');

func_chained = @chained_freudenstein_roth;

f1 = @(x) func_chained(x);
gradf1 = @(x) get_grad(func_chained, x);
Hessf1 = @(x) get_hess(func_chained, x);


func_broyden = @broyden_tridiagonal;

f2 = @(x) func_broyden(x);
gradf2 = @(x) get_grad_b(func_broyden, x);
Hessf2 = @(x) get_hess_b(func_broyden, x);

filename = fullfile(data_dir, 'dati.mat');
save(filename, 'f1', 'gradf1', 'Hessf1', 'f2', 'gradf2', 'Hessf2');

% funzioni di get
function g = get_grad(f_handle, x)
    [~, g] = f_handle(x);
end

function h = get_hess(f_handle, x)
    [~, ~, h] = f_handle(x);
end

% funzioni di get
function g = get_grad_b(f_handle, x)
    [~, g] = f_handle(x);
end

function h = get_hess_b(f_handle, x)
    [~, ~, h] = f_handle(x);
end


