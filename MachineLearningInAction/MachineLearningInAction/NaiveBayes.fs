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
        |> Seq.groupBy (fun m -> m.Value)
        |> Seq.map (fun (value, groups) -> value.ToLower(), (groups |> Seq.length))

    // Extracts all words used in a string.
    let vocabulary text =
        matchWords.Matches(text)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> m.Value.ToLower())
        |> Seq.distinct

    // Extracts all words used in a dataset;
    // a Dataset is a sequence of "samples", 
    // each sample has a label (the class), and text.
    let extractWords dataset =
        dataset 
        |> Seq.map (fun sample -> vocabulary (snd sample))
        |> Seq.concat
        |> Seq.distinct

    // "Tokenize" the dataset: break each text sample
    // into words and how many times they are used.
    let prepare dataset =
        dataset
        |> Seq.map (fun (label, sample) -> (label, wordsCount sample))