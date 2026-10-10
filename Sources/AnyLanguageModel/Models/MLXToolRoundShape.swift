#if MLX
    import MLXLMCommon

    /// What a model's chat template does with a tool round, read from what it renders.
    ///
    /// Two made-up conversations are rendered through the model's own chat template,
    /// without running the model, and four facts are read from the text.
    /// A render that fails leaves every fact that depends on it false.
    struct MLXToolRoundTraits: Equatable, Sendable {
        /// A round that replays the call ahead of its result renders at all.
        var rendersRound = false
        /// The call is printed, ahead of the result.
        var printsCall = false
        /// The result is printed.
        var printsResult = false
        /// Text beside the call is printed ahead of the call,
        /// and the prompt ends as it does without that text.
        var printsTextBeforeCall = false

        private static let userMarker = "zq7user"
        private static let argumentMarker = "zq7arg1"
        private static let resultMarker = "zq7res1"
        private static let textMarker = "zq7text"

        /// Renders two made-up conversations with `render` and reads the facts off the text.
        ///
        /// Both conversations hold a user message and one tool round built by `makeMLXToolRoundMessages`.
        /// The second differs from the first only in the text beside the call.
        /// A render that throws leaves every fact that depends on it false.
        static func probe(
            render: ([MLXLMCommon.Chat.Message]) async throws -> String
        ) async -> MLXToolRoundTraits {
            var traits = MLXToolRoundTraits()
            let call = MLXLMCommon.ToolCall(
                function: .init(name: "lookup", arguments: ["query": .string(argumentMarker)]),
                id: "call_probe1"
            )
            func conversation(text: String) -> [MLXLMCommon.Chat.Message] {
                var chat: [MLXLMCommon.Chat.Message] = [.user(userMarker)]
                chat += makeMLXToolRoundMessages(
                    calls: [call],
                    results: [resultMarker],
                    text: text,
                    shape: .withCalls(textWithFirstCall: true)
                )
                return chat
            }

            guard let plain = try? await render(conversation(text: "")) else { return traits }
            traits.rendersRound = true
            traits.printsResult = plain.contains(resultMarker)
            guard let callAt = plain.range(of: argumentMarker), let resultAt = plain.range(of: resultMarker),
                callAt.lowerBound < resultAt.lowerBound
            else { return traits }
            traits.printsCall = true

            guard let withText = try? await render(conversation(text: textMarker)) else { return traits }
            if let textAt = withText.range(of: textMarker), let callAtWithText = withText.range(of: argumentMarker),
                textAt.lowerBound < callAtWithText.lowerBound,
                let endOfPlain = promptEnd(of: plain), let endOfWithText = promptEnd(of: withText),
                endOfPlain == endOfWithText
            {
                traits.printsTextBeforeCall = true
            }
            return traits
        }

        /// The text after the last result, or `nil` when the result isn't printed.
        private static func promptEnd(of prompt: String) -> String? {
            prompt.range(of: resultMarker, options: .backwards).map { String(prompt[$0.upperBound...]) }
        }
    }

    /// How a finished tool round is fed back to the model.
    enum MLXToolRoundShape: Equatable, Sendable {
        /// The round's text as an assistant message when there is any, then each result as a tool message.
        case withoutCalls
        /// Every call in an assistant message of its own, directly followed by its result.
        /// The round's text is the content of the first call's message only where `textWithFirstCall` holds.
        case withCalls(textWithFirstCall: Bool)

        /// Picks the shape from what the model's chat template prints.
        ///
        /// Where the template prints both the call and the result, the round is fed back with its calls.
        /// Anything else, including a probe that failed, gives the results alone.
        init(_ traits: MLXToolRoundTraits) {
            if traits.rendersRound && traits.printsCall && traits.printsResult {
                self = .withCalls(textWithFirstCall: traits.printsTextBeforeCall)
            } else {
                self = .withoutCalls
            }
        }
    }
#endif  // MLX
