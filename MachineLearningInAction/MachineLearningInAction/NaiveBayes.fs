namespace MachineLearning

module NaiveBayes =

    open System
    open System.Text.RegularExpressions

    // Regular Expression matching full words, case insensitive.
    let matchWords = new Regex(@"\w+", RegexOptions.IgnoreCase)

    // Extract and count words from a string.
    // http://stackoverflow.com/a/2159085/114519        
    let wordsCount text =
        matchWords.Matches(text)
        |> Seq.cast<Match>
        |> Seq.groupBy (fun m -> m.Value.ToLower())
        |> Seq.map (fun (value, groups) -> 
            value.ToLower(), (groups |> Seq.length))
        |> Map.ofSeq

    // Extracts all words used in a string.
    let vocabulary text =
        matchWords.Matches(text)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> m.Value.ToLower())
        |> Set.ofSeq

    // Extracts all words used in a dataset;
    // a Dataset is a sequence of "samples", 
    // each sample has a label (the class), and text.
    let extractWords dataset =
        dataset 
        |> Seq.map (fun sample -> vocabulary (snd sample))
        |> Seq.concat
        |> Set.ofSeq

    // "Tokenize" the dataset: break each text sample
    // into words and how many times they are used.
    let prepare dataset =
        dataset
        |> Seq.map (fun (label, sample) -> (label, wordsCount sample))

    // Set-of-Words Accumulator function: 
    // state is the current count for each word so far, 
    // sample the tokenized text.
    // setFold increases the count by 1 if the word is 
    // present in the sample.
    let setFold state sample =
        state
        |> Map.map (fun token count ->
            if sample |> Map.containsKey(token) 
            then count + 1
            else count)

    // Bag-of-Words Accumulator function: 
    // state is the current count for each word so far, 
    // sample the tokenized text.
    // setFold increases the count by the number of occurences
    // of the word in the sample.
    let bagFold state sample =
        state
        |> Map.map (fun token count -> 
            if sample |> Map.containsKey(token)
            then count + sample.[token]
            else count)

    // Aggregate words frequency across the dataset,
    // using the provided folder.
    // (Supports setFold and bagFold)
    let frequency folder dataset words =
        let init = 
            words 
            |> Seq.map (fun w -> (w, 1))
            |> Map.ofSeq
        dataset
        |> Seq.fold (fun state (label, sample) -> folder state sample) init

    // Convenience functions for training the classifier
    // using set-of-Words and bag-of-Words frequency.
    let bagOfWords dataset words = frequency bagFold dataset words
    let setOfWords dataset words = frequency setFold dataset words

    // Converts 2 integers into a proportion.
    let prop (count, total) = (float)count / (float)total

    // Train based on a set of words and a dataset:
    // the dataset is "tokenized", and broken down into
    // one dataset per classification label.
    // For each group, we compute:
    // the proportion of the group relative to total,
    // the probability of each word within the group.
    let train frequency dataset words =
        let size = Seq.length dataset
        dataset
        |> prepare
        |> Seq.groupBy fst
        |> Seq.map (fun (label, data) -> 
            label, Seq.length data, frequency data words)
        |> Seq.map (fun (label, total, tokenCount) ->
            let totTokens = Map.fold (fun total key value -> total + value) 0 tokenCount
            label, 
            prop(total, size), 
            Map.map (fun token count -> prop(count, totTokens)) tokenCount)

    // Classifier function:
    // the classifier is trained on the dataset,
    // using the words and frequency folder supplied.
    // A piece of text is classified by computing
    // the "likelihood" it belongs to each possible label,
    // by checking the presence and weight of each
    // "classification word" in the tokenized text,
    // and returning the highest scoring label.
    // Probabilities are log-transformed to avoid underflow.
    // See "Chapter4.fsx" for an illustration.
    let classifier frequency dataset words text =
        let estimator = train frequency dataset words
        let tokenized = vocabulary text
        estimator
        |> Seq.map (fun (label, proba, tokens) ->
            label,
            tokens
            |> Map.fold (fun p token value -> 
                if Set.exists (fun w -> w = token) tokenized 
                then p + log(value) 
                else p) (log proba))
        |> Seq.maxBy snd
        |> fst