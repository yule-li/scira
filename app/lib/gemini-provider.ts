import { GoogleGenerativeAI } from '@google/generative-ai';
import { customProvider, wrapLanguageModel, extractReasoningMiddleware, LanguageModelV1, type LanguageModelV1CallOptions, type LanguageModelV1StreamPart } from 'ai';

export const gemini = (model: string) => {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
  const modelInstance = genAI.getGenerativeModel({ 
    model,
    generationConfig: {
      temperature: 0.7,
      topP: 0.8,
      topK: 40,
      maxOutputTokens: 2048,
    }
  });

  const wrappedModel: LanguageModelV1 = {
    provider: 'google',
    modelId: model,
    async doGenerate(options: LanguageModelV1CallOptions) {
      const result = await modelInstance.generateContent(options.prompt.toString());
      const text = result.response.text();
      return {
        text,
        toolCalls: [],
        finishReason: 'stop',
        usage: {
          promptTokens: 0,
          completionTokens: 0,
          totalTokens: 0
        },
        rawCall: {
          rawPrompt: options.prompt,
          rawSettings: {
            temperature: 0.7,
            topP: 0.8,
            topK: 40,
            maxOutputTokens: 2048
          }
        },
        rawResponse: {
          headers: {}
        },
        warnings: [],
        providerMetadata: {
          google: {
            safetyRatings: null
          }
        }
      };
    },
    async doStream(options: LanguageModelV1CallOptions) {
      const result = await modelInstance.generateContentStream(options.prompt.toString());
      const stream = result.stream;
      return new ReadableStream<LanguageModelV1StreamPart>({
        async start(controller) {
          try {
            for await (const chunk of stream) {
              const text = chunk.text();
              controller.enqueue({
                type: 'text-delta',
                textDelta: text
              });
            }
            controller.close();
          } catch (error) {
            controller.error(error);
          }
        }
      });
    }
  };

  return wrapLanguageModel({
    model: wrappedModel,
    middleware: extractReasoningMiddleware({ tagName: 'think' }),
  });
};
