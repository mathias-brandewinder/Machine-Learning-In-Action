#load "NaiveBayes.fs"
open MachineLearning.NaiveBayes

open System
open System.Text.RegularExpressions
   
let dataset =
    [| ("Ham",  "My dog has flea problems help please");
       ("Spam", "Maybe not take him to dog park stupid");
       ("Ham",  "My dalmatian is so cute I love him");
       ("Spam", "Stop posting stupid worthless garbage");
       ("Ham",  "Mr Licks ate my steak how to stop him");
       ("Spam", "Quit buying worthless dog food stupid") |]

// Retrieve all words from the dataset
let tokens = extractWords dataset

// Create 2 classifiers, using all the words found
let setClassifier = classifier setOfWords dataset tokens
let bagClassifier = classifier bagOfWords dataset tokens

// apply the set-of-words classifier 
// to all elements from the dataset,
// and retrieves actual and predicted labels
let setOfWordsTest =
    dataset
    |> Seq.map (fun t -> fst t, setClassifier (snd t))
    |> Seq.toList

// apply the bag-of-words classifier 
// to all elements from the dataset.
let bagOfWordsTest =
    dataset
    |> Seq.map (fun t -> fst t, bagClassifier (snd t))
    |> Seq.toList