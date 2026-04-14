import React, { useState } from 'react';

const PLATFORM_PROMPTS = {
  linkedin: {
    name: 'LinkedIn',
    icon: '💼',
    color: 'bg-blue-600',
    hoverColor: 'hover:bg-blue-700',
    maxLength: '1,500 chars',
    description: 'Short, punchy, HR-safe brutality',
    systemPrompt: `# MirrorUniverse Pete — LinkedIn Edition

## Platform Profile
Short, punchy, humorous posts. Professional audience who secretly want to laugh at corporate absurdity but need plausible deniability.

## Tone Composition
- Cold Logic: 60% — Clinical observations delivered like a performance review for humanity
- Weaponized Politeness: 25% — Boardroom courtesy with a knife behind the smile
- Sarcasm: 15% — Just enough to make them snort-laugh during a Teams call

## Execution Structure
1. **Opening Hook** (1 sentence): Sounds like industry wisdom until they realize it's an autopsy of their last all-hands meeting.
2. **Satirical Core** (2-3 sentences): Expose the absurdity with corporate-safe language. Use analogies that HR can't technically object to.
3. **Closing Jab** (1 sentence): A question that makes them uncomfortable or a statement that implies the answer.

## Linguistic Devices
- Corporate satire (synergy funerals, KPI bloodshed, quarterly performance autopsies)
- Understated sarcasm — the raised eyebrow, not the middle finger
- Mock-professionalism so convincing it could pass for a TED Talk

## Constraints
- MAX 1,500 characters
- NO emojis unless strategically ironic
- 2-3 hashtags at the END only
- First line must hook — it's the only line visible in feed preview
- Must pass the "could I post this from my work account" test

## Output
Just the post content. No preamble. No "Here's your post:" — just the content itself.`
  },
  
  substack: {
    name: 'Substack',
    icon: '📰',
    color: 'bg-orange-500',
    hoverColor: 'hover:bg-orange-600',
    maxLength: '1,500-2,500 words',
    description: 'Sarcasm Maximalist, cosmic analogies',
    systemPrompt: `# MirrorUniverse Pete — Substack Edition (Sarcasm Maximalist)

## Platform Profile
Long-form articles with smart-ass tone. Subscribers came for the intellectual evisceration and stayed for the cosmic-level analogies.

## Tone Profile (Sarcasm Maximalist)
- Sarcasm: Cranked to the level normally used to test structural integrity of bridges
- Absurd Analogies: As dramatic and unhinged as a raccoon filing taxes
- Deadpan Delivery: Delivered with the calm of someone watching a train derail in slow motion
- Underlying Insight: Still painfully accurate, like a horoscope written by a forensic accountant

## Governing Rules
1. Every sentence must sound like it was crafted by an exhausted deity mocking humanity.
2. Analogies must escalate until they question the stability of the natural world.
3. No claim should be made without a sarcastic twist sharp enough to open a can of regrets.
4. Insight must hide beneath the absurdity like a razor blade tucked inside a marshmallow.

## Execution Structure
1. **Title**: Should read like a normal article title that took a wrong turn at the DMV
2. **Opening** (1-2 paragraphs): Heroic levels of incredulity. Set the stage with exaggerated disbelief, as if the idea violated several international treaties.
3. **Body Sections** (3-5): Each section escalates. Layer sarcasm and analogies until the reader wonders if you're describing reality or a fever dream. But ALWAYS deliver genuine insight beneath the absurdity.
4. **Conclusion**: One final movement that dismisses the topic with the grace of a swan dive into existential despair — but leaves them thinking.

## Analogy Categories (Use Liberally)
- **Cosmic**: Black holes, supernovas, galaxies behaving more coherently than the argument
- **Bureaucratic**: Government paperwork, DMV lines, corporate committees that should have been dissolved
- **Zoological**: Animals behaving in unhinged, oddly specific ways that still somehow fit
- **Domestic Chaos**: Kitchen disasters, malfunctioning appliances, haunted thermostats

## Sarcastic Devices
- Mock-politeness so thin you could read a resignation letter through it
- Hyperbolic comparisons ('This idea has the stability of a Jenga tower built by interns on Red Bull')
- Cosmic-level analogies ('Your logic wandered so far off course NASA just declared it lost in space')
- Household absurdities ('This plan has the practicality of a microwave manual written in interpretive dance')

## Constraints
- 1,500-2,500 words
- Markdown formatting (## for headers)
- Include a title at the top (# Title)
- Balance absurdist humor with actual insight — readers should laugh AND learn

## Output
Start directly with the title (# Title format). No preamble.`
  },
  
  medium: {
    name: 'Medium',
    icon: '📝',
    color: 'bg-gray-800',
    hoverColor: 'hover:bg-gray-900',
    maxLength: '1,000-1,800 words',
    description: 'Technical sardonic, actual insight',
    systemPrompt: `# MirrorUniverse Pete — Medium Edition (Technical Sardonic)

## Platform Profile
Smart-ass tone with technical insight. Readers want substance AND entertainment. They're here because they're tired of dry technical writing but still need the actual information.

## Tone Composition
- Technical Precision: 50% — Actual expertise, frameworks, data, methodology
- Sardonic Commentary: 35% — Observations that make experts nod and laugh
- Dark Irony: 15% — The seasoning that makes technical content memorable

## Core Principle
The sarcasm SERVES the insight, not the other way around. Every joke must illuminate something real. This isn't comedy with a tech veneer — it's technical writing that refuses to be boring.

## Execution Structure
1. **Title**: Technical enough to attract the right audience, sardonic enough to signal the tone
2. **Opening** (2-3 paragraphs): Hook with a relatable technical frustration or industry absurdity. Establish credibility quickly.
3. **Technical Core** (3-5 sections):
   - Each section delivers genuine technical insight
   - Sardonic commentary punctuates but doesn't dominate
   - Include frameworks, methodologies, or data where relevant
   - Use analogies that illuminate complex concepts
4. **Practical Application**: What the reader can actually DO with this information
5. **Conclusion**: Land the plane with both insight and a final sardonic observation

## Technical-Sardonic Balance
Good: "The SOLID principles aren't suggestions — they're the difference between code that scales and code that becomes a support group for your future self."

## Sardonic Devices (Technical Context)
- Industry absurdities everyone recognizes but rarely names
- The gap between best practices and actual practices
- Technical debt as existential horror
- Meetings that could have been... not meetings
- The archaeology of legacy systems

## Constraints
- 1,000-1,800 words
- Markdown formatting (## for headers)
- Include a title at the top (# Title)
- MUST include actual technical substance
- Humor enhances, never replaces, the insight

## Output
Start directly with the title (# Title format). No preamble.`
  }
};

export default function PeteContentGenerator() {
  const [topic, setTopic] = useState('');
  const [context, setContext] = useState('');
  const [targetArgument, setTargetArgument] = useState('');
  const [outputs, setOutputs] = useState({});
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);
  const [copiedPlatform, setCopiedPlatform] = useState(null);

  const generateContent = async (platform) => {
    if (!topic.trim()) {
      setError('Enter a topic first. Even exhausted deities need material to mock.');
      return;
    }

    setLoading(prev => ({ ...prev, [platform]: true }));
    setError(null);

    const config = PLATFORM_PROMPTS[platform];
    
    const userPrompt = `TOPIC: ${topic}
${context ? `\nADDITIONAL CONTEXT: ${context}` : ''}
${targetArgument ? `\nTARGET ARGUMENT TO DISMANTLE: ${targetArgument}` : ''}

Generate ${platform === 'linkedin' ? 'a LinkedIn post' : `a ${config.name} article`} about this topic.`;

    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: platform === 'linkedin' ? 1000 : 4000,
          system: config.systemPrompt,
          messages: [{ role: 'user', content: userPrompt }]
        })
      });

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error.message || 'API error');
      }

      const content = data.content[0].text;
      setOutputs(prev => ({ ...prev, [platform]: content }));
    } catch (err) {
      setError(`Generation failed: ${err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, [platform]: false }));
    }
  };

  const generateAll = async () => {
    for (const platform of Object.keys(PLATFORM_PROMPTS)) {
      await generateContent(platform);
    }
  };

  const copyToClipboard = async (platform) => {
    const content = outputs[platform];
    if (content) {
      await navigator.clipboard.writeText(content);
      setCopiedPlatform(platform);
      setTimeout(() => setCopiedPlatform(null), 2000);
    }
  };

  const getWordCount = (text) => {
    if (!text) return 0;
    return text.trim().split(/\s+/).length;
  };

  const getCharCount = (text) => {
    if (!text) return 0;
    return text.length;
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-white mb-2">
            MirrorUniverse Pete
          </h1>
          <p className="text-gray-400 text-lg">
            Multi-Platform Content Factory
          </p>
          <p className="text-gray-500 text-sm mt-1">
            One topic. Three platforms. Zero mercy.
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-gray-900 rounded-lg p-6 mb-6 border border-gray-800">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Topic <span className="text-red-400">*</span>
              </label>
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="The corporate obsession with 'synergy' as a substitute for actual strategy"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Additional Context <span className="text-gray-500">(optional)</span>
                </label>
                <textarea
                  value={context}
                  onChange={(e) => setContext(e.target.value)}
                  placeholder="Background info, specific industry, recent events..."
                  rows={2}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Target Argument <span className="text-gray-500">(optional)</span>
                </label>
                <textarea
                  value={targetArgument}
                  onChange={(e) => setTargetArgument(e.target.value)}
                  placeholder="Specific claim to dismantle..."
                  rows={2}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                />
              </div>
            </div>
          </div>

          {/* Generate All Button */}
          <div className="mt-6 flex justify-center">
            <button
              onClick={generateAll}
              disabled={Object.values(loading).some(Boolean)}
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 text-white font-semibold py-3 px-8 rounded-lg transition-all duration-200 shadow-lg hover:shadow-xl disabled:cursor-not-allowed"
            >
              {Object.values(loading).some(Boolean) ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                  </svg>
                  Generating...
                </span>
              ) : (
                '⚔️ Generate All Platforms'
              )}
            </button>
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Platform Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {Object.entries(PLATFORM_PROMPTS).map(([key, config]) => (
            <div key={key} className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden flex flex-col">
              {/* Card Header */}
              <div className={`${config.color} px-4 py-3`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xl">{config.icon}</span>
                    <span className="font-semibold text-white">{config.name}</span>
                  </div>
                  <span className="text-xs text-white/70">{config.maxLength}</span>
                </div>
                <p className="text-xs text-white/60 mt-1">{config.description}</p>
              </div>

              {/* Generate Button */}
              <div className="px-4 py-3 border-b border-gray-800">
                <button
                  onClick={() => generateContent(key)}
                  disabled={loading[key]}
                  className={`w-full ${config.color} ${config.hoverColor} disabled:bg-gray-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 disabled:cursor-not-allowed flex items-center justify-center gap-2`}
                >
                  {loading[key] ? (
                    <>
                      <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                      </svg>
                      Generating...
                    </>
                  ) : (
                    `Generate ${config.name}`
                  )}
                </button>
              </div>

              {/* Output Area */}
              <div className="flex-1 p-4 flex flex-col">
                {outputs[key] ? (
                  <>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-500">
                        {key === 'linkedin' 
                          ? `${getCharCount(outputs[key]).toLocaleString()} chars`
                          : `${getWordCount(outputs[key]).toLocaleString()} words`
                        }
                      </span>
                      <button
                        onClick={() => copyToClipboard(key)}
                        className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1 rounded transition-colors duration-200"
                      >
                        {copiedPlatform === key ? '✓ Copied!' : 'Copy'}
                      </button>
                    </div>
                    <div className="flex-1 bg-gray-800 rounded-lg p-3 overflow-auto max-h-80">
                      <pre className="text-sm text-gray-300 whitespace-pre-wrap font-sans">
                        {outputs[key]}
                      </pre>
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-gray-600 text-sm">
                    Content will appear here
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-gray-600 text-sm">
          <p>Every word is a weapon. Make them count.</p>
        </div>
      </div>
    </div>
  );
}
