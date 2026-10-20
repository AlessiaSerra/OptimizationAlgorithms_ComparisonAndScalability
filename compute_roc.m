function p = compute_roc(xseq, xbest)
    % COMPUTE_ROC Stima p usando la formula algebrica su 3 punti.
    % Nessuna regressione lineare.
    
         
    % 1. Calcola l'errore reale (distanza da xbest)
    err = sqrt(sum((xseq - xbest).^2, 1)); 
    
    % 2. Filtra i valori troppo vicini allo zero macchina
    %    Sotto 1e-14 i logaritmi diventano instabili.
    %    (Nota: se xbest è perfetto, l'ultimo err potrebbe essere 0 -> log(-Inf))
    err = err(err > 1e-15);
    
    % Dopo il filtro, ci servono ancora almeno 3 punti (k, k+1, k+2)
    if length(err) < 3
        p = NaN;
        return;
    end
    
    % 3. Prendi gli ULTIMI 3 errori validi
    %    e_k   = err(end-2)
    %    e_k1  = err(end-1)  (cioè k+1)
    %    e_k2  = err(end)    (cioè k+2)
    
    e_k   = err(end-2);
    e_k1  = err(end-1);
    e_k2  = err(end);
    
    % 4. Calcola i logaritmi dei rapporti (numeratore e denominatore)
    %    Formula: p = log(e_{k+2}/e_{k+1}) / log(e_{k+1}/e_k)
    
    log_ratio_next = log(e_k2 / e_k1);
    log_ratio_prev = log(e_k1 / e_k);
    
    % 5. Gestione divisione per zero
    %    Se l'errore non cambia (stallo), log_ratio_prev è 0.
    if abs(log_ratio_prev) < eps
        p = NaN; 
        return;
    end
    
    % 6. Calcolo finale
    p = log_ratio_next / log_ratio_prev;
    
    % 7. Filtri di sanità (opzionale ma consigliato)
    %    Valori negativi o enormi indicano solitamente errori numerici
    if p < -1 || p > 10
        p = NaN;
    end
end