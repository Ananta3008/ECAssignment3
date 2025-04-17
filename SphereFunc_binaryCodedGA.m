function binary_ga_sphere()
    % Parameters (as shown in pseudocode)
    fitness_function = @sphere_function;  % Objective function
    n_var = 4;                % Number of variables
    lb = -10;                 % Lower bound
    ub = 10;                  % Upper bound
    Np = 50;                  % Population size
    T = 50;                  % Number of generations
    bits_per_var = 16;        % Precision (bits per variable)
    nD = n_var * bits_per_var;% Total chromosome length
    pc = 0.8;                 % Crossover probability
    pm = 0.05;               % Mutation probability
    k = 3;                    % Tournament size
    
    % Initialize random population (P) of binary strings
    P = randi([0 1], Np, nD);
    
    % Decode binary population to real values
    decoded_pop = decode_population(P, n_var, bits_per_var, lb, ub);
    
    % Evaluate initial fitness
    fitness = zeros(Np, 1);
    for i = 1:Np
        fitness(i) = fitness_function(decoded_pop(i,:));
    end
    
    % Track best solution
    [best_fitness, best_idx] = min(fitness);
    best_solution = decoded_pop(best_idx,:);
    best_fitness_history = zeros(T, 1);
    
    % Main loop
    for t = 1:T
        % Create empty offspring population
        offspring = zeros(Np, nD);
        offspring_fitness = zeros(Np, 1);
        
        % Create offspring through selection, crossover and mutation
        for i = 1:Np/2
            % Tournament selection to select two parents
            parent1_idx = tournament_selection(fitness, k);
            parent2_idx = tournament_selection(fitness, k);
            parent1 = P(parent1_idx,:);
            parent2 = P(parent2_idx,:);
            
            % Crossover (with probability pc)
            if rand < pc
                % Select crossover site
                crossover_site = randi([1 nD-1]);
                % Single-point crossover
                offspring1 = [parent1(1:crossover_site) parent2(crossover_site+1:end)];
                offspring2 = [parent2(1:crossover_site) parent1(crossover_site+1:end)];
            else
                % Copy parents
                offspring1 = parent1;
                offspring2 = parent2;
            end
            
            % Store in offspring population
            offspring(2*i-1,:) = offspring1;
            offspring(2*i,:) = offspring2;
        end
        
        % Mutation
        for i = 1:Np
            % Generate nD random numbers between 0 and 1
            r = rand(1, nD);
            % Perform bit-wise mutation with probability pm
            mutation_sites = r < pm;
            offspring(i, mutation_sites) = 1 - offspring(i, mutation_sites);
        end
        
        % Decode offspring to real values
        decoded_offspring = decode_population(offspring, n_var, bits_per_var, lb, ub);
        
        % Evaluate fitness of offspring
        for i = 1:Np
            offspring_fitness(i) = fitness_function(decoded_offspring(i,:));
        end
        
        % Combine population and offspring and select best Np individuals (μ + λ)
        combined_pop = [P; offspring];
        combined_decoded = [decoded_pop; decoded_offspring];
        combined_fitness = [fitness; offspring_fitness];
        
        % Sort by fitness (ascending for minimization)
        [sorted_fitness, sorted_idx] = sort(combined_fitness);
        
        % Select the best Np individuals
        P = combined_pop(sorted_idx(1:Np), :);
        decoded_pop = combined_decoded(sorted_idx(1:Np), :);
        fitness = sorted_fitness(1:Np);
        
        % Update best solution
        if fitness(1) < best_fitness
            best_fitness = fitness(1);
            best_solution = decoded_pop(1,:);
        end
        
        % Store best fitness for plotting
        best_fitness_history(t) = best_fitness;
        
        % Display progress every 10 generations
        if mod(t, 10) == 0
            fprintf('Generation %d: Best Fitness = %.6f\n', t, best_fitness);
        end
    end
    
    % Display final results
    fprintf('\nOptimization complete.\n');
    fprintf('Best solution found: [%.6f, %.6f, %.6f, %.6f]\n', best_solution);
    fprintf('Fitness value: %.6f\n', best_fitness);
    
    % Plot convergence
    figure;
    plot(1:T, best_fitness_history, 'LineWidth', 2);
    xlabel('Generation');
    ylabel('Best Fitness');
    title('Convergence Curve');
    grid on;
end

function value = sphere_function(x)
    % Sphere function: f(x) = sum(x.^2) for all variables
    value = sum(x.^2);
end

function idx = tournament_selection(fitness, k)
    % Tournament selection
    % Randomly select k individuals and return the best one
    pop_size = length(fitness);
    tournament_indices = randi(pop_size, 1, k);
    tournament_fitness = fitness(tournament_indices);
    [~, best_local_idx] = min(tournament_fitness); % min for minimization
    idx = tournament_indices(best_local_idx);
end

function decoded = decode_population(pop, n_var, bits_per_var, lb, ub)
    % Decode binary population to real values
    [pop_size, chromosome_length] = size(pop);
    decoded = zeros(pop_size, n_var);
    
    for i = 1:pop_size
        for j = 1:n_var
            % Extract bits for variable j
            start_bit = (j-1) * bits_per_var + 1;
            end_bit = j * bits_per_var;
            binary_var = pop(i, start_bit:end_bit);
            
            % Convert binary to decimal
            decimal_value = bin2dec(num2str(binary_var));
            
            % Scale to range [lb, ub]
            max_decimal = 2^bits_per_var - 1;
            decoded(i, j) = lb + (decimal_value / max_decimal) * (ub - lb);
        end
    end
end