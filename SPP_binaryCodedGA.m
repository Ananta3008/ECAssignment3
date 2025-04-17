function SPP_binaryCodedGA()
    solve_large_gap_ga();
end

function solve_large_gap_ga()
    output_file = 'gap_binary_ga.txt'; % Renamed output file
    fid_out = fopen(output_file, 'w');  % Open file for writing
    
    if fid_out == -1
        error('Unable to create output file.');
    end

    % Iterate through gap1 to gap12 dataset files
    for g = 1:12
        filename = sprintf('gap%d.txt', g);
        fid = fopen(filename, 'r');
        if fid == -1
            error('Error opening file %s.', filename);
        end
        
        % Read the number of problem sets
        num_problems = fscanf(fid, '%d', 1);

        % Display and write dataset name (gapX)
        dataset_title = filename(1:end-4); % Removes .txt
        fprintf('\n%s\n', dataset_title);
        fprintf(fid_out, '\n%s\n', dataset_title);
        
        for p = 1:num_problems
            % Read problem parameters
            m = fscanf(fid, '%d', 1); % Number of servers
            n = fscanf(fid, '%d', 1); % Number of users
            
            % Read cost and resource matrices
            c = fscanf(fid, '%d', [n, m])';
            r = fscanf(fid, '%d', [n, m])';
            
            % Read server capacities
            b = fscanf(fid, '%d', [m, 1]);
            
            % Solve using Genetic Algorithm (GA)
            x_matrix = solve_gap_ga(m, n, c, r, b);
            objective_value = sum(sum(c .* x_matrix)); % Maximization
            
            % Format: c{m}{n}-{instance number} result
            result_line = sprintf('c%d-%d %d', m*100 + n, p, round(objective_value));
            
            % Print to console
            fprintf('%s\n', result_line);
            
            % Write to file
            fprintf(fid_out, '%s\n', result_line);
        end
        
        % Close dataset file
        fclose(fid);
    end
    
    % Close result output file
    fclose(fid_out);
    fprintf('\nAll results written to %s\n', output_file);
end

function x_matrix = solve_gap_ga(m, n, c, r, b)
    % GA Parameters
    pop_size = 100;
    max_gen = 300;
    crossover_rate = 0.8;
    mutation_rate = 0.09;

    % Initialize and make feasible
    population = zeros(pop_size, m * n);
    for i = 1:pop_size
        population(i, :) = enforce_feasibility(rand(1, m * n), m, n);
    end
    
    % Evaluate initial fitness
    fitness = arrayfun(@(i) fitnessFcn(population(i, :)), 1:pop_size);
    
    for gen = 1:max_gen
        % Selection
        parents = tournamentSelection(population, fitness);
        
        % Crossover
        offspring = singlePointCrossover(parents, crossover_rate);
        
        % Mutation
        mutated_offspring = mutation(offspring, mutation_rate);
        
        % Make offspring feasible
        for i = 1:size(mutated_offspring, 1)
            mutated_offspring(i, :) = enforce_feasibility(mutated_offspring(i, :), m, n);
        end
        
        % Evaluate new fitness
        new_fitness = arrayfun(@(i) fitnessFcn(mutated_offspring(i, :)), 1:size(mutated_offspring, 1));
        
        % Combine populations
        [~, best_idx] = max([fitness, new_fitness]);
        if best_idx > length(fitness)
            population = mutated_offspring;
            fitness = new_fitness;
        else
            population = [population; mutated_offspring];
            fitness = [fitness, new_fitness];
        end
        
        % Keep best individuals
        [~, sorted_idx] = sort(fitness, 'descend');
        population = population(sorted_idx(1:pop_size), :);
        fitness = fitness(sorted_idx(1:pop_size));
    end
    
    % Return best
    [~, best_idx] = max(fitness);
    x_matrix = reshape(population(best_idx, :), [m, n]);

    % Fitness with penalties
    function fval = fitnessFcn(x)
        x_mat = reshape(x, [m, n]);
        cost = sum(sum(c .* x_mat));
        cap_violation = sum(max(sum(x_mat .* r, 2) - b, 0));
        assign_violation = sum(abs(sum(x_mat, 1) - 1));
        penalty = 1e6 * (cap_violation + assign_violation);
        fval = cost - penalty;
    end
end

function selected = tournamentSelection(population, fitness)
    pop_size = size(population, 1);
    selected = zeros(size(population));
    
    for i = 1:pop_size
        idx1 = randi(pop_size);
        idx2 = randi(pop_size);
        if fitness(idx1) > fitness(idx2)
            selected(i, :) = population(idx1, :);
        else
            selected(i, :) = population(idx2, :);
        end
    end
end

function offspring = singlePointCrossover(parents, crossover_rate)
    pop_size = size(parents, 1);
    num_genes = size(parents, 2);
    offspring = parents;
    
    for i = 1:2:pop_size-1
        if rand < crossover_rate
            point = randi(num_genes - 1);
            offspring(i, point+1:end) = parents(i+1, point+1:end);
            offspring(i+1, point+1:end) = parents(i, point+1:end);
        end
    end
end

function mutated = mutation(offspring, mutation_rate)
    mutated = offspring;
    for i = 1:numel(offspring)
        if rand < mutation_rate
            mutated(i) = 1 - mutated(i); % Flip bit
        end
    end
end

function x_corrected = enforce_feasibility(x, m, n)
    x_mat = reshape(x, [m, n]);
    for j = 1:n
        [~, idx] = max(x_mat(:, j));
        x_mat(:, j) = 0;
        x_mat(idx, j) = 1;
    end
    x_corrected = reshape(x_mat, [1, m * n]);
end
