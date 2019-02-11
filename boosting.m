clear;clc;close all;


% load specified dataset
set = 'pima';


% set reference to training function
if strcmpi(set,'car')
    trainboost = @carBoost;
    bestc = 40;
    bests = 40;
    
elseif strcmpi(set,'pima')
    trainboost = @pimaBoost;
    bestc = 30;
    bests = 10;
    
else
    error('Please use either the ''adult'' set or the ''other'' set');
    
end

splits = [1,10:10:100];
learn = [0.1,0.5,0.9];
cycles = [1,10:10:100];

[train,test] = loadData(set,4641);
for s = 1:length(splits)
    for l = 1:length(learn)
        for c = 1:length(cycles)
            [currmodel,curracc,currtime] = trainboost(train,splits(s),cycles(c),learn(l));

            boostres(s,c,l).model = currmodel;
            boostres(s,c,l).traintime =  currtime;
            boostres(s,c,l).valacc = curracc;
            boostres(s,c,l).trainacc = sum(currmodel.predictFcn(train) == train{:,1})/size(train,1);
            tic
            boostres(s,c,l).testacc = sum(currmodel.predictFcn(test) == test{:,1})/size(test,1);
            boostres(s,c,l).testtime = toc;
        end
    end
end


for train_percentage = 1:20
   tr_idx = size(train,1)*(train_percentage*5/100);
   numdata(train_percentage) = round(tr_idx);
   for p = 1:10
       train = train(randperm(length(train{:,1})),:);
       train_limit = train(1:round(tr_idx),:);

       [currmodel,data_amt_valacc(train_percentage,p),currtime] = trainboost(train_limit,bests,bestc,0.1);

       data_amt_testacc(train_percentage,p) = sum(currmodel.predictFcn(test) == test{:,1})/size(test,1);
       data_amt_trainacc(train_percentage,p) = sum(currmodel.predictFcn(train_limit) == train_limit{:,1})/size(train_limit,1);
   end
end

figure
plot(numdata,mean(data_amt_valacc,2))
hold on
plot(numdata,mean(data_amt_testacc,2))
plot(numdata,mean(data_amt_trainacc,2))
grid minor
legend('Validation accuracy','Test accuracy','Training accuracy')
xlabel('Amount of training data')
ylabel('Accuracy')


for q = 1:l
    valsurf = reshape([boostres(:,:,q).valacc],[size(boostres,1),size(boostres,2)]);
    trainsurf = reshape([boostres(:,:,q).trainacc],[size(boostres,1),size(boostres,2)]);
    testsurf = reshape([boostres(:,:,q).testacc],[size(boostres,1),size(boostres,2)]);
    
    figure
    hold on
    surf(splits,cycles,valsurf,'FaceAlpha',0.7,'FaceColor','r');
    surf(splits,cycles,testsurf,'FaceAlpha',0.7,'FaceColor','b');
    surf(splits,cycles,trainsurf,'FaceAlpha',0.7,'FaceColor','y');
    grid minor
    ylabel('Maximum splits');
    xlabel('Iterations');
    zlabel('Accuracy');
    legend('Validation','Test','Train');
    title(['Learning rate = ',num2str(learn(q))]);
end

save(['boost_',set,'.mat'],'boostres','valsurf','testsurf','trainsurf');
