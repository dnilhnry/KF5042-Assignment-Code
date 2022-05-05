clc; clear;

truePos = 0;
trueNeg = 0;
falsePos = 0;
falseNeg = 0;

reviewData = readtable("testData.csv");
reviewText = reviewData.review;
actualScore = reviewData.score;
prepData = preprocessReviews(reviewText);

rng('default');
embeddings = fastTextWordEmbedding;

fidPositive = fopen(fullfile('opinion-lexicon-English','positive-words.txt'));
C = textscan(fidPositive,'%s','CommentStyle',';');
positiveWords = string(C{1});

fidNegative = fopen(fullfile('opinion-lexicon-English','negative-words.txt'));
C = textscan(fidNegative,'%s','CommentStyle',';');
negativeWords = string(C{1});
fclose all;

words = [positiveWords; negativeWords];
labels = categorical(nan(numel(words),1));
labels( 1: numel(positiveWords)) = "Positive";
labels( numel(positiveWords) + 1 : end ) = "Negative";
data = table(words, labels, 'VariableNames', {'Word' , 'Label'});
wordsToRemove = ~isVocabularyWord(embeddings, data.Word);
data(wordsToRemove,:) = [];

numWords = size(data,1);
partition = cvpartition(numWords,'HoldOut',0.05);
trainingData = data(training(partition),:);
testingData = data(test(partition),:);

trainingWords = trainingData.Word;
XTrain = word2vec(embeddings, trainingWords);
YTrain = trainingData.Label;

SVM = fitcsvm(XTrain,YTrain);

testingWords = testingData.Word;
XTest = word2vec(embeddings,testingWords);
YTest = testingData.Label;
[YPrediction, scores] = predict(SVM,XTest);
figure
confusionchart(YTest,YPrediction);

sentimentScores = zeros(size(prepData));

for i = 1 : prepData.length
    data = prepData(i).Vocabulary;
    vector = word2vec(embeddings, data);
    [ ~ , scores ] = predict(SVM, vector );
    sentimentScores(i) = mean(scores(:,1));
    if (isnan(sentimentScores(i)))
        sentimentScores(i) = 0;
    end

    if (sentimentScores(i) <= -0.75)
        sentimentScores(i) = 1;
    elseif ( (sentimentScores(i) >= -0.75) && (sentimentScores(i) < -0.25) )
        sentimentScores(i) = 2;
    elseif ( (sentimentScores(i) >= -0.25) && (sentimentScores(i) < 0.25) )
        sentimentScores(i) = 3;
    elseif ( (sentimentScores(i) >= 0.25) && (sentimentScores(i) < 0.75) )
        sentimentScores(i) = 4;
    elseif (sentimentScores(i) < 0.75)
        sentimentScores(i) = 5;
    end

    if ((sentimentScores(i) == 5) && (actualScore(i) == 5))
        truePos = truePos + 1;
    elseif ((sentimentScores(i) == 4) && (actualScore(i) == 4))
        truePos = truePos + 1;

    elseif ((sentimentScores(i) == 1) && (actualScore(i) == 1))
        trueNeg = trueNeg + 1;
    elseif ((sentimentScores(i) == 2) && (actualScore(i) == 2))
        trueNeg = trueNeg + 1;
    
    elseif ((sentimentScores(i) > 3) && (actualScore(i) < 3))
        falsePos = falsePos + 1;
    
    elseif ((sentimentScores(i) < 3) && (actualScore(i) > 3))
        falseNeg = falseNeg +1;
    end
    fprintf('Sent: %d | words: %s | Found Score: %d | Correct Score: %d \n', i, joinWords(prepData(i)), sentimentScores(i), actualScore(i));
end

NoOfNeutral = sum(sentimentScores == 3);
NoOfFound = numel(sentimentScores) - NoOfNeutral;

fprintf("Coverage: %2.2f%% | Number Found: %d | Neutral: %d \n", (NoOfFound*100)/numel(sentimentScores), NoOfFound, NoOfNeutral);
fprintf("true pos: %d | true neg: %d | false pos: %d | false neg: %d",truePos,trueNeg,falsePos,falseNeg);