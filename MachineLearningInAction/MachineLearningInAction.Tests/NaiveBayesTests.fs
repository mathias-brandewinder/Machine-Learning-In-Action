namespace MachineLearning.Tests

open MachineLearning.NaiveBayes
open NUnit.Framework
open FsUnit

[<TestFixture>]
type ``Naive Bayes tests`` () =

    let text = "I call it my billion-dollar mistake. It was the invention of the null reference in 1965."

    [<TestCase("i", Result = 1)>]
    [<TestCase("call", Result = 1)>]
    [<TestCase("it", Result = 2)>]
    [<TestCase("the", Result = 2)>]
    [<TestCase("billion", Result = 1)>]
    [<TestCase("dollar", Result = 1)>]
    [<TestCase("1965", Result = 1)>]
    member this.``wordsCount verification`` (word) =
        
        (wordsCount text).[word]

    [<TestCase("i", Result = true)>]
    [<TestCase("call", Result = true)>]
    [<TestCase("it", Result = true)>]
    [<TestCase("trumpet", Result = false)>]
    [<TestCase("It", Result = false)>]
    member this.``vocabulary verification`` (word) =
        
        (vocabulary text).Contains(word)

    [<Test>]
    member this.``setFold verification`` () =

        let acc    = Map.empty.Add("A", 0).Add("B", 0).Add("C", 1).Add("D", 1)
        let sample = Map.empty.Add("B", 1).Add("C", 42)
        let folded = setFold acc sample

        folded.["A"] |> should equal 0
        folded.["B"] |> should equal 1
        folded.["C"] |> should equal 2
        folded.["D"] |> should equal 1
    
    [<Test>]
    member this.``bagFold verification`` () =

        let acc    = Map.empty.Add("A", 0).Add("B", 0).Add("C", 1).Add("D", 1)
        let sample = Map.empty.Add("B", 1).Add("C", 42)
        let folded = bagFold acc sample

        folded.["A"] |> should equal 0
        folded.["B"] |> should equal 1
        folded.["C"] |> should equal 43
        folded.["D"] |> should equal 1
