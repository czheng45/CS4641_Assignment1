function [train,test] = loadData(name,rn)
    rng(rn);    
    trainp = 0.8;


    
    if strcmpi(name,'car')
        load([cd,'\data\car.mat'],'data');
        
    elseif strcmpi(name,'pima')
        load([cd,'\data\pima.mat'],'data');
        
    else
        error('Please choose dataset ''pima'' or dataset ''car''.');
        
    end
    
    idx = round(trainp*size(data,1));
    data = data(randperm(length(data{:,1})),:);

    % Withold 20% of the data for testing after training

    train = data(1:idx,:); % 80% is train set
    test = data(idx+1:end,:); % 20% is test set
end

