open System
open System.IO
open MachineLearning.NaiveBayes
open Newtonsoft.Json
open Newtonsoft.Json.Linq

let main =

    let stackoverflow = "stackoverflow"
    let programmers = "programmers"

    let extractFromJson text =
        let json = JsonConvert.DeserializeObject<JObject>(text);
        let titles = 
            json.["items"] :?> JArray
            |> Seq.map (fun item -> item.["title"].ToString())
        titles

    let extractFromFile file = File.ReadAllText(file)

    let dataset = seq {
            yield! extractFromFile("StackOverflowTraining.txt") 
                |> extractFromJson 
                |> Seq.take 50
                |> Seq.map (fun t -> stackoverflow, t)
            yield! extractFromFile("ProgrammersTraining.txt") 
                |> extractFromJson 
                |> Seq.take 50
                |> Seq.map (fun t -> programmers, t)
        }

    printfn "Training the classifier"

    // http://www.textfixer.com/resources/common-english-words.txt
    let stopWords = "a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your"
    let remove = stopWords.Split(',')

    let words = 
        dataset
        |> extractWords
        |> Seq.filter (fun w -> 
            Array.exists (fun e -> e = w) remove |> not) 

    let classify = classifier bagOfWords dataset words

    let testset = seq {
            yield! extractFromFile("StackOverflowTest.txt") 
                |> extractFromJson 
                |> Seq.map (fun t -> stackoverflow, t)
            yield! extractFromFile("ProgrammersTest.txt") 
                |> extractFromJson 
                |> Seq.map (fun t -> programmers, t)
        }

    printfn "Classifying"  
    testset |> Seq.iter (fun sample -> 
        printfn "%s, %s" (fst sample) (classify (snd sample)))

    Console.ReadKey()