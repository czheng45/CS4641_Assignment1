clear;clc;close all;


% load specified dataset
set = 'car';



mxlayers = 3;
mxperceptrons = 30;


if strcmpi(set,'car')
    
    bestl = 1;
    bestp = 20;
 
    for ly = 1:mxlayers
        for p = 1:mxperceptrons
            for r = 1:10
                [traindata,testdata] = loadData(set,r*4641);

                traintarget = dummyvar(traindata{:,1})';
                testtarget = dummyvar(testdata{:,1})';
                intrain = [];
                intest = [];
                for q = 2:size(traindata,2)
                    intrain = [dummyvar(traindata{:,q}),intrain];
                    intest = [dummyvar(testdata{:,q}),intest];
                end
                intrain = intrain';
                intest = intest';
                
                net = patternnet(repmat(p,[1,ly]));
                tic
                [net,tr] = train(net,intrain,traintarget);
                nets{r} = net;
                trainTime(r) = toc;

                results = round(net(intrain));
                trainacc(r) = sum(results(1,:) == traintarget(1,:))/size(traintarget,2);
                
                testacc(r) = 1 - tr.best_tperf;
                valacc(r) = 1 - tr.best_vperf;
            end
            nnet(ly,p).trainTime = mean(trainTime);
            nnet(ly,p).nnet = nets;
            nnet(ly,p).trainacc = mean(trainacc);
            nnet(ly,p).testacc = mean(testacc);
            nnet(ly,p).valacc = mean(valacc);
        end
    end
    
elseif strcmpi(set,'pima')

    bestl = 1;
    bestp = 16;
    
    for ly = 1:mxlayers
        for p = 1:mxperceptrons
            for r = 1:10
                [traindata,testdata] = loadData(set,r*4641);

                traintarget = dummyvar(traindata{:,1}+1)';
                testtarget = dummyvar(testdata{:,1}+1)';
                intrain = traindata{:,2:end}';
                intest = testdata{:,2:end}';


                net = patternnet(repmat(p,[1,ly]));
                tic
                [net,tr] = train(net,intrain,traintarget);
                nets{r} = net;
                trainTime(r) = toc;

                results = round(net(intrain));
                trainacc(r) = sum(results(1,:) == traintarget(1,:))/size(traintarget,2);
                
                testacc(r) = 1 - tr.best_tperf;
                valacc(r) = 1 - tr.best_vperf;
            end
            nnet(ly,p).trainTime = mean(trainTime);
            nnet(ly,p).nnet = nets;
            nnet(ly,p).trainacc = mean(trainacc);
            nnet(ly,p).testacc = mean(testacc);
            nnet(ly,p).valacc = mean(valacc);
        end
    end
    
else
    error('Please use either the ''car'' set or the ''pima'' set');
end


intrain = intrain';
intest = intest';
traintarget = traintarget';

for train_percentage = 1:20
   tr_idx = size(intrain,1)*(train_percentage*5/100);
   numdata(train_percentage) = round(tr_idx);
   for p = 1:10
       swapidx = randperm(length(intrain(:,1)));
       intrain = intrain(swapidx,:);
       traintarget = traintarget(swapidx,:);
       train_limit = intrain(1:round(tr_idx),:)';
       train_limit_target = traintarget(1:round(tr_idx),:)';
       
       net = patternnet(repmat(bestp,[1,bestl]));
       [net,tr] = train(net,train_limit,train_limit_target);

       data_amt_valacc(train_percentage,p) = 1 - tr.best_vperf;
       data_amt_testacc(train_percentage,p) = 1 - tr.best_tperf;
       
       results = round(net(train_limit));    
       data_amt_trainacc(train_percentage,p) = sum(results(1,:) == train_limit_target(1,:))/size(train_limit_target,2);
       
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

trainacc = reshape([nnet.trainacc],[mxlayers,mxperceptrons]);
valacc = reshape([nnet.valacc],[mxlayers,mxperceptrons]);
testacc = reshape([nnet.testacc],[mxlayers,mxperceptrons]);

figure
hold on
plot(trainacc(1,:),'--r')
plot(valacc(1,:),'r')
plot(testacc(1,:),':r')
plot(trainacc(2,:),'--b')
plot(valacc(2,:),'b')
plot(testacc(2,:),':b')
plot(trainacc(3,:),'--k')
plot(valacc(3,:),'k')
plot(testacc(3,:),':k')
grid minor

legend('1L train','1L validate','1L test','2L train','2L validate','2L test','3L train','3L validate','3L test','location','northwestoutside');

xlabel('Number of perceptrons per layer')
ylabel('Accuracy')

save(['nnet_',set,'.mat'],'nnet','mxlayers','mxperceptrons');

