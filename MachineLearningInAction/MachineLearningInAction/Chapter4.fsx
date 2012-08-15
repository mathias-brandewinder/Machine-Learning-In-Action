open System
open System.Text.RegularExpressions

let words = new Regex(@"\w+", RegexOptions.IgnoreCase)

// http://stackoverflow.com/a/2159085/114519        
let wordsCount string =
    words.Matches(string)
    |> Seq.cast<Match>
    |> Seq.groupBy (fun m -> m.Value)
    |> Seq.map (fun (value, groups) -> value.ToLower(), (groups |> Seq.length))

let vocabulary string =
    words.Matches(string)
    |> Seq.cast<Match>
    |> Seq.map (fun m -> m.Value)
    |> Seq.distinct
   
let dataset =
    [| ("Ham",  "My dog has flea problems help please");
       ("Spam", "Maybe not take him to dog park stupid");
       ("Ham",  "My dalmatian is so cute I love him");
       ("Spam", "Stop posting stupid worthless garbage");
       ("Ham",  "Mr Licks ate my steak how to stop him");
       ("Spam", "Quit buying worthless dog food stupid") |]

let prepare dataset =
    dataset
    |> Seq.map (fun (label, sample) -> (label, wordsCount sample))

let update state sample =
    state
    |> Seq.map (fun (token, count) -> 
        if Seq.exists (fun (t, c) -> t = token) sample 
        then (token, count + 1.0) 
        else (token, count))

let estimate dataset words =
    let init = words |> Seq.map (fun w -> (w, 0.0))
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