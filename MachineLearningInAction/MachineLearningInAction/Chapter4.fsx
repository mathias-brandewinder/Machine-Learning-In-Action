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

let update state sample =
    state
    |> Seq.map (fun (token, count) -> 
        if Seq.exists (fun (t, c) -> t = token) sample 
        then (token, count + 1.0) 
        else (token, count))

let estimate dataset words =
    let init = words |> Seq.map (fun w -> (w, 1.0))
    dataset
    |> Seq.fold (fun state (label, sample) -> update state sample) init

let evaluate dataset words =
    let size = Seq.length dataset
    dataset
    |> prepare
    |> Seq.groupBy fst
    |> Seq.map (fun (label, data) -> label, Seq.length data, estimate data words)
    |> Seq.map (fun (label, total, tokenCount) ->
        let tokensTotal = Seq.sumBy (fun t -> snd t) tokenCount
        label, (float)total/(float)size, Seq.map (fun (token, count) -> token, count / tokensTotal) tokenCount)

let classify dataset words text =
    let estimator = evaluate dataset words
    let tokenized = vocabulary text
    estimator
    |> Seq.map (fun (label, proba, tokens) ->
        label,
        tokens
        |> Seq.fold (fun p token -> 
            if Seq.exists (fun w -> w = fst token) tokenized 
            then p + log(snd token) 
            else p) (log proba))
    |> Seq.maxBy snd
    |> fst

let testWords = extractWords dataset
let classifier = classify dataset testWords
let test = Seq.map (fun s -> snd s) dataset |> Seq.map (fun t -> classifier t) |> Seq.toList