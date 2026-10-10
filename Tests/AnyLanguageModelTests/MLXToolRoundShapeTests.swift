import Foundation
import Testing

@testable import AnyLanguageModel

#if MLX
    import Jinja
    import enum MLXLMCommon.Chat
    import struct MLXLMCommon.DefaultMessageGenerator
    import struct MLXLMCommon.ToolCall
    import struct MLXVLM.Gemma4MessageGenerator
    import struct MLXVLM.Qwen3VLMessageGenerator

    private typealias RawMessage = [String: any Sendable]

    /// The four facts as letters, in the order rendersRound, printsCall, printsResult, printsTextBeforeCall.
    extension MLXToolRoundTraits {
        fileprivate var letters: String {
            [rendersRound, printsCall, printsResult, printsTextBeforeCall].map { $0 ? "T" : "F" }.joined()
        }
    }

    /// The message form a model family's processor gives its chat template.
    private enum MessageFamily {
        case plain
        case qwen3VL
        case gemma4

        func generate(_ chat: [Chat.Message]) -> [RawMessage] {
            switch self {
            case .plain: DefaultMessageGenerator().generate(messages: chat)
            case .qwen3VL: Qwen3VLMessageGenerator().generate(messages: chat)
            case .gemma4: Gemma4MessageGenerator().generate(messages: chat)
            }
        }
    }

    /// Renders a chat template the way the tokenizer library does:
    /// the same engine options, `messages`, `add_generation_prompt`, the start-of-text token, and `tools` only when declared.
    private struct TemplateRenderer {
        let template: Jinja.Template
        let family: MessageFamily
        let startOfText: String?

        init(source: String, family: MessageFamily = .plain, startOfText: String? = nil) throws {
            template = try Jinja.Template(source, with: .init(lstripBlocks: true, trimBlocks: true))
            self.family = family
            self.startOfText = startOfText
        }

        func render(
            _ chat: [Chat.Message],
            tools: [RawMessage]? = nil,
            additionalContext: RawMessage? = nil
        ) throws -> String {
            var context: [String: Jinja.Value] = [
                "messages": .array(try family.generate(chat).map { try Jinja.Value(any: $0) }),
                "add_generation_prompt": .boolean(true),
            ]
            if let tools {
                context["tools"] = .array(try tools.map { try Jinja.Value(any: $0) })
            }
            if let startOfText {
                context["bos_token"] = .string(startOfText)
            }
            for (key, value) in additionalContext ?? [:] {
                context[key] = try Jinja.Value(any: value)
            }
            return try template.render(context)
        }
    }

    @Suite("MLX tool round shape")
    struct MLXToolRoundShapeTests {
        // MARK: Templates of our own

        private static let printsEverything = #"""
            {%- for message in messages -%}
            <|{{ message.role }}|>{{ message.content }}
            {%- if message.tool_calls -%}
            {%- for tool_call in message.tool_calls -%}
            <call>{{ tool_call.function.name }} {{ tool_call.function.arguments | tojson }}</call>
            {%- endfor -%}
            {%- endif -%}
            <|end|>
            {%- endfor -%}
            <|assistant|>
            """#

        private static let neverReadsCalls = #"""
            {%- for message in messages -%}
            <|{{ message.role }}|>{{ message.content }}<|end|>
            {%- endfor -%}
            <|assistant|>
            """#

        private static let rejectsTheRound = #"""
            {%- for message in messages -%}
            {%- if message.role == 'tool' -%}
            {{- raise_exception('This template takes no tool messages.') -}}
            {%- endif -%}
            <|{{ message.role }}|>{{ message.content }}<|end|>
            {%- endfor -%}
            <|assistant|>
            """#

        private static let printsTheCallAfterTheResult = #"""
            {%- for message in messages -%}
            <|{{ message.role }}|>{{ message.content }}<|end|>
            {%- endfor -%}
            {%- for message in messages -%}
            {%- if message.tool_calls -%}
            {%- for tool_call in message.tool_calls -%}
            <called>{{ tool_call.function.name }} {{ tool_call.function.arguments | tojson }}</called>
            {%- endfor -%}
            {%- endif -%}
            {%- endfor -%}
            <|assistant|>
            """#

        private static let dropsTheResult = #"""
            {%- for message in messages -%}
            {%- if message.role != 'tool' -%}
            <|{{ message.role }}|>{{ message.content }}
            {%- if message.tool_calls -%}
            {%- for tool_call in message.tool_calls -%}
            <call>{{ tool_call.function.name }} {{ tool_call.function.arguments | tojson }}</call>
            {%- endfor -%}
            {%- endif -%}
            <|end|>
            {%- endif -%}
            {%- endfor -%}
            <|assistant|>
            """#

        private static let printsTheTextAfterTheCall = #"""
            {%- for message in messages -%}
            <|{{ message.role }}|>
            {%- if message.tool_calls -%}
            {%- for tool_call in message.tool_calls -%}
            <call>{{ tool_call.function.name }} {{ tool_call.function.arguments | tojson }}</call>
            {%- endfor -%}
            {%- endif -%}
            {{ message.content }}<|end|>
            {%- endfor -%}
            <|assistant|>
            """#

        private static let changesTheEndForText = #"""
            {%- set state = namespace(spoke=false) -%}
            {%- for message in messages -%}
            <|{{ message.role }}|>{{ message.content }}
            {%- if message.tool_calls -%}
            {%- if message.content -%}
            {%- set state.spoke = true -%}
            {%- endif -%}
            {%- for tool_call in message.tool_calls -%}
            <call>{{ tool_call.function.name }} {{ tool_call.function.arguments | tojson }}</call>
            {%- endfor -%}
            {%- endif -%}
            <|end|>
            {%- endfor -%}
            {%- if not state.spoke -%}
            <|assistant|>
            {%- endif -%}
            """#

        /// The traits the probe finds when it renders with `source`.
        private func traits(ofTemplate source: String) async throws -> MLXToolRoundTraits {
            let renderer = try TemplateRenderer(source: source)
            return await MLXToolRoundTraits.probe { chat in try renderer.render(chat) }
        }

        @Test func syntheticTemplatePrintsEverything() async throws {
            let found = try await traits(ofTemplate: Self.printsEverything)
            #expect(found.letters == "TTTT")
        }

        @Test func syntheticTemplateNeverReadsCalls() async throws {
            let found = try await traits(ofTemplate: Self.neverReadsCalls)
            #expect(found.letters == "TFTF")
        }

        @Test func syntheticTemplateRejectsTheRound() async throws {
            let found = try await traits(ofTemplate: Self.rejectsTheRound)
            #expect(found.letters == "FFFF")
        }

        @Test func syntheticTemplatePrintsTheCallAfterTheResult() async throws {
            let found = try await traits(ofTemplate: Self.printsTheCallAfterTheResult)
            #expect(found.letters == "TFTF")
        }

        @Test func syntheticTemplateDropsTheResult() async throws {
            let found = try await traits(ofTemplate: Self.dropsTheResult)
            #expect(found.letters == "TFFF")
        }

        @Test func syntheticTemplatePrintsTheTextAfterTheCall() async throws {
            let found = try await traits(ofTemplate: Self.printsTheTextAfterTheCall)
            #expect(found.letters == "TTTF")
        }

        @Test func syntheticTemplateChangesTheEndForText() async throws {
            let found = try await traits(ofTemplate: Self.changesTheEndForText)
            #expect(found.letters == "TTTF")
        }

        // MARK: The table

        @Test func shapeFollowsTheTraits() {
            // rendersRound, printsCall, printsResult, printsTextBeforeCall, and the shape the table gives.
            let rows: [(Bool, Bool, Bool, Bool, MLXToolRoundShape)] = [
                (false, false, false, false, .withoutCalls),
                (false, false, false, true, .withoutCalls),
                (false, false, true, false, .withoutCalls),
                (false, false, true, true, .withoutCalls),
                (false, true, false, false, .withoutCalls),
                (false, true, false, true, .withoutCalls),
                (false, true, true, false, .withoutCalls),
                (false, true, true, true, .withoutCalls),
                (true, false, false, false, .withoutCalls),
                (true, false, false, true, .withoutCalls),
                (true, false, true, false, .withoutCalls),
                (true, false, true, true, .withoutCalls),
                (true, true, false, false, .withoutCalls),
                (true, true, false, true, .withoutCalls),
                (true, true, true, false, .withCalls(textWithFirstCall: false)),
                (true, true, true, true, .withCalls(textWithFirstCall: true)),
            ]
            for (renders, call, result, text, expected) in rows {
                let traits = MLXToolRoundTraits(
                    rendersRound: renders,
                    printsCall: call,
                    printsResult: result,
                    printsTextBeforeCall: text
                )
                print("shape of \(traits.letters): \(MLXToolRoundShape(traits))")
                #expect(MLXToolRoundShape(traits) == expected)
            }
        }

        // MARK: A probe that fails

        private struct RenderFailed: Error {}

        @Test func aRenderThatThrowsGivesNoTraits() async {
            let found = await MLXToolRoundTraits.probe { _ in throw RenderFailed() }

            #expect(found == MLXToolRoundTraits())
            #expect(found.letters == "FFFF")
            #expect(MLXToolRoundShape(found) == .withoutCalls)
        }

        @Test func aSecondRenderThatThrowsLeavesTheTextOut() async {
            var renders = 0
            let found = await MLXToolRoundTraits.probe { _ in
                renders += 1
                if renders > 1 { throw RenderFailed() }
                return "zq7user zq7arg1 zq7res1"
            }

            #expect(renders == 2)
            #expect(found.letters == "TTTF")
            #expect(MLXToolRoundShape(found) == .withCalls(textWithFirstCall: false))
        }

        // MARK: What the probe renders

        @Test func probeRendersTheRoundTheBuilderMakes() async throws {
            var conversations: [[RawMessage]] = []
            _ = await MLXToolRoundTraits.probe { chat in
                conversations.append(DefaultMessageGenerator().generate(messages: chat))
                return "zq7arg1 zq7res1"
            }

            try #require(conversations.count == 2)
            for (index, conversation) in conversations.enumerated() {
                #expect(conversation.map { $0["role"] as? String } == ["user", "assistant", "tool"])
                let assistant = conversation[1]
                #expect(assistant["content"] as? String == (index == 0 ? "" : "zq7text"))
                let calls = try #require(assistant["tool_calls"] as? [[String: any Sendable]])
                #expect(calls.count == 1)
                #expect(calls.first?["id"] as? String == "call_probe1")
                let function = try #require(calls.first?["function"] as? [String: any Sendable])
                #expect(function["name"] as? String == "lookup")
                let arguments = try #require(function["arguments"] as? [String: any Sendable])
                #expect(arguments["query"] as? String == "zq7arg1")
                #expect(conversation[2]["tool_call_id"] as? String == "call_probe1")
                #expect(conversation[2]["name"] as? String == "lookup")
                #expect(conversation[2]["content"] as? String == "zq7res1")
            }
            // The two conversations differ in the text beside the call and in nothing else.
            #expect(conversations[0][0]["content"] as? String == conversations[1][0]["content"] as? String)
            #expect(conversations[0][2]["content"] as? String == conversations[1][2]["content"] as? String)
        }

        // MARK: Published templates

        /// A published chat template, its size, how its model family builds messages,
        /// its start-of-text token, and the facts recorded for it.
        private struct PublishedTemplate {
            let file: String
            let bytes: Int
            let family: MessageFamily
            let startOfText: String?
            let letters: String
        }

        private static let published: [PublishedTemplate] = [
            PublishedTemplate(
                file: "qwen3-2507-instruct.jinja",
                bytes: 4040,
                family: .plain,
                startOfText: nil,
                letters: "TTTT"
            ),
            PublishedTemplate(
                file: "qwen3.5.jinja",
                bytes: 7756,
                family: .qwen3VL,
                startOfText: nil,
                letters: "TTTT"
            ),
            PublishedTemplate(
                file: "gemma-4-e.jinja",
                bytes: 17336,
                family: .gemma4,
                startOfText: "<bos>",
                letters: "TTTF"
            ),
            PublishedTemplate(
                file: "lfm2.5-conversion.jinja",
                bytes: 1836,
                family: .plain,
                startOfText: "<|startoftext|>",
                letters: "TFTF"
            ),
            PublishedTemplate(
                file: "lfm2.5-liquid.jinja",
                bytes: 5487,
                family: .plain,
                startOfText: "<|startoftext|>",
                letters: "TTTT"
            ),
            PublishedTemplate(
                file: "gemma-3-text.jinja",
                bytes: 1532,
                family: .plain,
                startOfText: "<bos>",
                letters: "FFFF"
            ),
        ]

        static let templatesFolder = ProcessInfo.processInfo.environment["MLX_TOOL_ROUND_TEMPLATES"]

        private static let lookupTool: RawMessage = [
            "type": "function",
            "function": [
                "name": "lookup",
                "description": "Looks something up.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "query": ["type": "string", "description": "What to look up."] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "required": ["query"] as [String],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]

        private func position(_ needle: String, in text: String) -> Int? {
            text.range(of: needle).map { text.distance(from: text.startIndex, to: $0.lowerBound) }
        }

        private func promptEnd(of text: String, after marker: String) -> String? {
            text.range(of: marker, options: .backwards).map { String(text[$0.upperBound...]) }
        }

        @Test(
            .enabled(
                if: MLXToolRoundShapeTests.templatesFolder != nil,
                "MLX_TOOL_ROUND_TEMPLATES does not name a folder of chat templates"
            )
        )
        func publishedTemplatesHaveTheirRecordedTraits() async throws {
            let folder = try #require(Self.templatesFolder)
            for template in Self.published {
                let url = URL(fileURLWithPath: folder).appendingPathComponent(template.file)
                let data = try Data(contentsOf: url)
                #expect(data.count == template.bytes, "\(template.file) has \(data.count) bytes, not \(template.bytes)")
                let source = try #require(String(data: data, encoding: .utf8))
                let renderer = try TemplateRenderer(
                    source: source,
                    family: template.family,
                    startOfText: template.startOfText
                )

                // One tool declared or none, the thinking flag absent, true or false, a system message or none.
                for tools in [nil, [Self.lookupTool]] as [[RawMessage]?] {
                    for thinking in [nil, true, false] as [Bool?] {
                        for system in [false, true] {
                            let context: RawMessage? = thinking.map { (value: Bool) -> RawMessage in
                                ["enable_thinking": value]
                            }
                            let found = await MLXToolRoundTraits.probe { chat in
                                try renderer.render(
                                    (system ? [.system("zq7system")] : []) + chat,
                                    tools: tools,
                                    additionalContext: context
                                )
                            }
                            #expect(
                                found.letters == template.letters,
                                "\(template.file), tools \(tools != nil), thinking \(String(describing: thinking)), system \(system)"
                            )
                        }
                    }
                }

                // Where the text may stand beside a call, it also stands before the first call of a round of two.
                if template.letters.last == "T" {
                    func twoCalls(text: String) -> [Chat.Message] {
                        let first = ToolCall(
                            function: .init(name: "lookup", arguments: ["query": .string("zq7arg1")]),
                            id: "call_probe1"
                        )
                        let second = ToolCall(
                            function: .init(name: "lookup", arguments: ["query": .string("zq7arg2")]),
                            id: "call_probe2"
                        )
                        return [.user("zq7user")]
                            + makeMLXToolRoundMessages(
                                calls: [first, second],
                                results: ["zq7res1", "zq7res2"],
                                text: text,
                                shape: .withCalls(textWithFirstCall: true)
                            )
                    }
                    let plain = try renderer.render(twoCalls(text: ""))
                    let withText = try renderer.render(twoCalls(text: "zq7text"))
                    let positions = ["zq7text", "zq7arg1", "zq7res1", "zq7arg2", "zq7res2"].map {
                        position($0, in: withText)
                    }
                    #expect(positions.allSatisfy { $0 != nil }, "\(template.file): a marker is missing: \(positions)")
                    #expect(
                        positions.compactMap { $0 } == positions.compactMap { $0 }.sorted(),
                        "\(template.file): markers out of order"
                    )
                    let endOfPlain = promptEnd(of: plain, after: "zq7res2")
                    #expect(endOfPlain != nil)
                    #expect(
                        endOfPlain == promptEnd(of: withText, after: "zq7res2"),
                        "\(template.file): the text changes the prompt end"
                    )
                }

                let base = await MLXToolRoundTraits.probe { chat in try renderer.render(chat) }
                print("tool-round traits: \(template.file) \(base.letters)")
            }
        }

        // MARK: Loaded models

        static let modelsSpec = ProcessInfo.processInfo.environment["MLX_TOOL_ROUND_MODELS"]

        @Test(
            .enabled(
                if: MLXToolRoundShapeTests.modelsSpec != nil,
                "MLX_TOOL_ROUND_MODELS does not name model folders"
            )
        )
        func loadedModelsHaveTheTraitsOfTheirTemplates() async throws {
            let spec = try #require(Self.modelsSpec)
            let reportPath = ProcessInfo.processInfo.environment["MLX_TOOL_ROUND_REPORT"]
            for pair in spec.split(separator: ";") {
                let parts = pair.split(separator: "=", maxSplits: 1).map(String.init)
                guard parts.count == 2 else {
                    Issue.record("Not a folder=LETTERS pair: \(pair)")
                    continue
                }
                let folder = parts[0]
                let expected = parts[1]
                let url = URL(fileURLWithPath: folder)
                let model = MLXLanguageModel(modelId: url.lastPathComponent, directory: url)

                let line: String
                do {
                    let found = try await model.toolRoundTraits().letters
                    line = "tool-round traits: \(folder) found \(found) expected \(expected)"
                    #expect(found == expected, "\(folder)")
                } catch {
                    line = "tool-round traits: \(folder) found nothing (\(error)) expected \(expected)"
                    Issue.record("\(folder) did not load: \(error)")
                }
                print(line)
                if let reportPath {
                    let handle: FileHandle
                    if let existing = FileHandle(forWritingAtPath: reportPath) {
                        handle = existing
                    } else {
                        FileManager.default.createFile(atPath: reportPath, contents: nil)
                        handle = try FileHandle(forWritingTo: URL(fileURLWithPath: reportPath))
                    }
                    try handle.seekToEnd()
                    try handle.write(contentsOf: Data((line + "\n").utf8))
                    try handle.close()
                }
                // One model in memory at a time.
                await model.removeFromCache()
            }
        }
    }
#endif  // MLX
