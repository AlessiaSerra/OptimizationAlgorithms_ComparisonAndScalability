function [F, g, Hess] = broyden_tridiagonal(x)
    
    n = length(x);
    
    x_prev = [0; x(1:n-1)]; % Rappresenta x_{k-1}
    x_next = [x(2:n); 0];   % Rappresenta x_{k+1}

    f = (3 - 2*x) .* x - x_prev - 2.*x_next + 1;

    F = 0.5 * sum(f.^2);

    if nargout > 1
        d = 3 - 4*x;
        l = -ones(n, 1);
        u = -2 * ones(n, 1);
        % Creiamo J come matrice sparsa per efficienza
        J = spdiags([l d u], -1:1, n, n);
    end

    % 4. Calcolo del Gradiente g = J' * f
    if nargout >= 2
        % Esplicitando J' * f per componenti:
        % g_k = -2*f_{k-1} + (3-4x_k)*f_k - f_{k+1}
        f_prev = [0; f(1:n-1)];
        f_next = [f(2:n); 0];
        g = (3 - 4*x) .* f - 2*f_prev - f_next;
    end
    
    % 5. Calcolo dell'Hessiana Hess = J' * J + sum(f_k * nabla^2 f_k)
    if nargout == 3
        % Poiché la derivata seconda di f_k rispetto a x_k è -4 (e 0 altrove),
        % il termine di curvatura è una matrice diagonale: diag(-4 * f)
        curvature_term = spdiags(-4 * f, 0, n, n);
        Hess = J' * J + curvature_term;
    end
end