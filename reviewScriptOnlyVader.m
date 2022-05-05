clc; clear;

truePos = 0;
trueNeg = 0;
falsePos = 0;
falseNeg = 0;

reviewData = readtable("testData.csv");
reviewText = reviewData.review;
actualScore = reviewData.score;
prepData = preprocessReviews(reviewText);

sentimentScores = zeros(size(prepData));

sentimentScores = vaderSentimentScores(prepData);
for i = 1 : prepData.length
    
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

NumNeutral = sum(sentimentScores == 3);
NumFound = numel(sentimentScores) - NumNeutral;

fprintf("Coverage: %2.2f%% | Number Found: %d | Neutral: %d \n", (NumFound*100)/numel(sentimentScores), NumFound, NumNeutral);
fprintf("true pos: %d | true neg: %d | false pos: %d | false neg: %d",truePos,trueNeg,falsePos,falseNeg);