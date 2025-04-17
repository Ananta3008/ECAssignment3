function convergence_graph_BinaryGA()
    output = {};
    
    for dataset = 12
        file = sprintf('gap%d.txt', dataset);
        fid = fopen(file, 'r');
        if fid == -1
            error('Cannot open file: %s', file);
        end

        problemCount = fscanf(fid, '%d', 1);
        fprintf('\n%s\n', erase(file, '.txt'));

        for instance = 1:problemCount
            numMachines = fscanf(fid, '%d', 1);
            numTasks = fscanf(fid, '%d', 1);
            profitMatrix = fscanf(fid, '%d', [numTasks, numMachines])';
            demandMatrix = fscanf(fid, '%d', [numTasks, numMachines])';
            capacityVector = fscanf(fid, '%d', [numMachines, 1]);

            showConvergence = (dataset == 12);
            [solution, bestHistory] = geneticAssignmentSolver(numMachines, numTasks, profitMatrix, demandMatrix, capacityVector, showConvergence);

            totalProfit = sum(sum(profitMatrix .* solution));
            label = sprintf('c%d-%d', numMachines * 100 + numTasks, instance);
            fprintf('%s  %d\n', label, round(totalProfit));
            output{end+1, 1} = label;
            output{end, 2} = round(totalProfit);

            if showConvergence
                figure;
                plot(1:length(bestHistory), bestHistory, '-s', 'LineWidth', 2);
                title(sprintf('Convergence Graph (Bianry GA) - %s (Problem %d)', erase(file, '.txt'), instance));
                xlabel('Generation');
                ylabel('Best Fitness');
                grid on;
            end
        end
        fclose(fid);
    end

    

   
end

function [assignmentMatrix, fitnessTrack] = geneticAssignmentSolver(machines, tasks, valueMatrix, reqMatrix, limits, trackProgress)
    populationSize = 100;
    maxGenerations = 100;
    crossRate = 0.8;
    mutateRate = 0.02;

    fitnessTrack = zeros(1, maxGenerations);

    individuals = generateInitialPopulation(populationSize, machines, tasks);
    scores = evaluatePopulation(individuals, machines, tasks, valueMatrix, reqMatrix, limits);

    for generation = 1:maxGenerations
        selected = selectParents(individuals, scores);
        children = applyCrossover(selected, crossRate);
        mutants = applyMutation(children, mutateRate);
        mutants = repairPopulation(mutants, machines, tasks);

        newScores = evaluatePopulation(mutants, machines, tasks, valueMatrix, reqMatrix, limits);

        combined = [individuals; mutants];
        combinedScores = [scores, newScores];

        [~, sortedIdx] = sort(combinedScores, 'descend');
        individuals = combined(sortedIdx(1:populationSize), :);
        scores = combinedScores(sortedIdx(1:populationSize));

        if trackProgress
            fitnessTrack(generation) = scores(1);
        end
    end

    [~, topIdx] = max(scores);
    assignmentMatrix = reshape(individuals(topIdx, :), [machines, tasks]);
end

function population = generateInitialPopulation(count, m, n)
    population = zeros(count, m * n);
    for i = 1:count
        randInd = rand(m * n, 1);
        population(i, :) = enforceConstraints(randInd, m, n);
    end
end

function scores = evaluatePopulation(pop, m, n, values, requirements, capacity)
    num = size(pop, 1);
    scores = zeros(1, num);
    for i = 1:num
        scores(i) = computeFitness(pop(i, :), m, n, values, requirements, capacity);
    end
end

function score = computeFitness(chrom, m, n, values, reqs, caps)
    mat = reshape(chrom, [m, n]);
    profit = sum(sum(values .* mat));
    overload = sum(max(sum(mat .* reqs, 2) - caps, 0));
    unassigned = sum(abs(sum(mat, 1) - 1));
    penalty = 1e6 * (overload + unassigned);
    score = profit - penalty;
end

function parents = selectParents(pop, fit)
    n = size(pop, 1);
    parents = zeros(size(pop));
    for i = 1:n
        a = randi(n);
        b = randi(n);
        parents(i, :) = pop(a, :) * (fit(a) >= fit(b)) + pop(b, :) * (fit(b) > fit(a));
    end
end

function newGen = applyCrossover(parents, rate)
    n = size(parents, 1);
    len = size(parents, 2);
    newGen = parents;
    for i = 1:2:n-1
        if rand < rate
            pt = randi(len - 1);
            newGen(i, pt+1:end) = parents(i+1, pt+1:end);
            newGen(i+1, pt+1:end) = parents(i, pt+1:end);
        end
    end
end

function mutatedGen = applyMutation(pop, rate)
    mutatedGen = pop;
    for i = 1:numel(pop)
        if rand < rate
            mutatedGen(i) = 1 - pop(i);
        end
    end
end

function repaired = repairPopulation(pop, m, n)
    repaired = zeros(size(pop));
    for i = 1:size(pop, 1)
        repaired(i, :) = enforceConstraints(pop(i, :), m, n);
    end
end

function valid = enforceConstraints(chromosome, m, n)
    reshaped = reshape(chromosome, [m, n]);
    for j = 1:n
        [~, idx] = max(reshaped(:, j));
        reshaped(:, j) = 0;
        reshaped(idx, j) = 1;
    end
    valid = reshape(reshaped, [1, m * n]);
end