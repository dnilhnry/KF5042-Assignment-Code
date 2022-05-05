function [documents] = preprocessReviews(textData)
cleanTextData = lower(textData);
documents = tokenizedDocument(cleanTextData);
documents = erasePunctuation(documents);
documents = removeStopWords(documents);
end
