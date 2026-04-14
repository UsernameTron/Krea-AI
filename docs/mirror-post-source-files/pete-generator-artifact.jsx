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
2. **Opening** (1-2 paragraphs): Heroic levels of incredulity.
3. **Body Sections** (3-5): Each section escalates. Layer sarcasm and analogies until the reader wonders if you're describing reality or a fever dream. But ALWAYS deliver genuine insight beneath the absurdity.
4. **Conclusion**: One final movement that dismisses the topic with the grace of a swan dive into existential despair — but leaves them thinking.

## Analogy Categories (Use Liberally)
- **Cosmic**: Black holes, supernovas, galaxies behaving more coherently than the argument
- **Bureaucratic**: Government paperwork, DMV lines, corporate committees that should have been dissolved
- **Zoological**: Animals behaving in unhinged, oddly specific ways that still somehow fit
- **Domestic Chaos**: Kitchen disasters, malfunctioning appliances, haunted thermostats

## Constraints
- 1,500-2,500 words
- Markdown formatting (## for headers)
- Include a title at the top (# Title)
- Balance absurdist humor with actual insight

## Output
Start directly with the title (# Title format). No preamble.`
  },
  
  medium: {
    name: 'Medium',
    icon: '📝',
    color: 'bg-gray-700',
    hoverColor: 'hover:bg-gray-800',
    maxLength: '1,000-1,800 words',
    description: 'Technical sardonic, actual insight',
    systemPrompt: `# MirrorUniverse Pete — Medium Edition (Technical Sardonic)

## Platform Profile
Smart-ass tone with technical insight. Readers want substance AND entertainment.

## Tone Composition
- Technical Precision: 50% — Actual expertise, frameworks, data, methodology
- Sardonic Commentary: 35% — Observations that make experts nod and laugh
- Dark Irony: 15% — The seasoning that makes technical content memorable

## Core Principle
The sarcasm SERVES the insight, not the other way around. Every joke must illuminate something real.

## Execution Structure
1. **Title**: Technical enough to attract the right audience, sardonic enough to signal the tone
2. **Opening** (2-3 paragraphs): Hook with a relatable technical frustration or industry absurdity.
3. **Technical Core** (3-5 sections): Each section delivers genuine technical insight with sardonic commentary.
4. **Practical Application**: What the reader can actually DO with this information
5. **Conclusion**: Land the plane with both insight and a final sardonic observation

## Sardonic Devices (Technical Context)
- Industry absurdities everyone recognizes but rarely names
- The gap between best practices and actual practices
- Technical debt as existential horror
- Meetings that could have been... not meetings

## Constraints
- 1,000-1,800 words
- Markdown formatting (## for headers)
- Include a title at the top (# Title)
- MUST include actual technical substance

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
  const [expandedPlatform, setExpandedPlatform] = useState(null);

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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: platform === 'linkedin' ? 1000 : 4000,
          system: config.systemPrompt,
          messages: [{ role: 'user', content: userPrompt }]
        })
      });

      const data = await response.json();
      if (data.error) throw new Error(data.error.message || 'API error');
      setOutputs(prev => ({ ...prev, [platform]: data.content[0].text }));
    } catch (err) {
      setError(`Generation failed: ${err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, [platform]: false }));
    }
  };

  const generateAll = async () => {
    if (!topic.trim()) {
      setError('Enter a topic first.');
      return;
    }
    for (const platform of Object.keys(PLATFORM_PROMPTS)) {
      await generateContent(platform);
    }
  };

  const copyToClipboard = async (platform) => {
    if (outputs[platform]) {
      await navigator.clipboard.writeText(outputs[platform]);
      setCopiedPlatform(platform);
      setTimeout(() => setCopiedPlatform(null), 2000);
    }
  };

  const getWordCount = (text) => text ? text.trim().split(/\s+/).length : 0;
  const getCharCount = (text) => text ? text.length : 0;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-6 text-center">
          <h1 className="text-2xl font-bold text-white mb-1">⚔️ MirrorUniverse Pete</h1>
          <p className="text-slate-400 text-sm">One topic. Three platforms. Zero mercy.</p>
        </div>

        {/* Input Section */}
        <div className="bg-slate-900 rounded-lg p-4 mb-4 border border-slate-800">
          <div className="space-y-3">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Topic *</label>
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="The corporate obsession with 'synergy' as a substitute for strategy"
                className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
              />
            </div>
            
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Context (optional)</label>
                <input
                  value={context}
                  onChange={(e) => setContext(e.target.value)}
                  placeholder="Background, industry, events..."
                  className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Target Argument (optional)</label>
                <input
                  value={targetArgument}
                  onChange={(e) => setTargetArgument(e.target.value)}
                  placeholder="Specific claim to dismantle..."
                  className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
                />
              </div>
            </div>
          </div>

          <div className="mt-4 flex justify-center">
            <button
              onClick={generateAll}
              disabled={Object.values(loading).some(Boolean)}
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-slate-600 disabled:to-slate-600 text-white font-medium py-2 px-6 rounded-lg transition-all text-sm disabled:cursor-not-allowed"
            >
              {Object.values(loading).some(Boolean) ? '⏳ Generating...' : '⚔️ Generate All Platforms'}
            </button>
          </div>

          {error && (
            <div className="mt-3 p-2 bg-red-900/50 border border-red-700 rounded text-red-200 text-xs">
              {error}
            </div>
          )}
        </div>

        {/* Platform Cards */}
        <div className="space-y-3">
          {Object.entries(PLATFORM_PROMPTS).map(([key, config]) => (
            <div key={key} className="bg-slate-900 rounded-lg border border-slate-800 overflow-hidden">
              {/* Header Row */}
              <div className={`${config.color} px-4 py-2 flex items-center justify-between`}>
                <div className="flex items-center gap-2">
                  <span>{config.icon}</span>
                  <span className="font-medium text-white text-sm">{config.name}</span>
                  <span className="text-xs text-white/60">({config.maxLength})</span>
                </div>
                <div className="flex items-center gap-2">
                  {outputs[key] && (
                    <span className="text-xs text-white/70">
                      {key === 'linkedin' ? `${getCharCount(outputs[key])} chars` : `${getWordCount(outputs[key])} words`}
                    </span>
                  )}
                  <button
                    onClick={() => generateContent(key)}
                    disabled={loading[key]}
                    className="bg-white/20 hover:bg-white/30 disabled:bg-white/10 text-white text-xs px-3 py-1 rounded transition-colors disabled:cursor-not-allowed"
                  >
                    {loading[key] ? '⏳' : 'Generate'}
                  </button>
                  {outputs[key] && (
                    <button
                      onClick={() => copyToClipboard(key)}
                      className="bg-white/20 hover:bg-white/30 text-white text-xs px-3 py-1 rounded transition-colors"
                    >
                      {copiedPlatform === key ? '✓' : 'Copy'}
                    </button>
                  )}
                </div>
              </div>

              {/* Content Area */}
              {outputs[key] && (
                <div className="p-3">
                  <div 
                    className={`bg-slate-800 rounded p-3 overflow-auto transition-all duration-200 ${
                      expandedPlatform === key ? 'max-h-96' : 'max-h-32'
                    }`}
                  >
                    <pre className="text-xs text-slate-300 whitespace-pre-wrap font-sans leading-relaxed">
                      {outputs[key]}
                    </pre>
                  </div>
                  <button
                    onClick={() => setExpandedPlatform(expandedPlatform === key ? null : key)}
                    className="mt-2 text-xs text-slate-500 hover:text-slate-300"
                  >
                    {expandedPlatform === key ? '▲ Collapse' : '▼ Expand'}
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="mt-6 text-center text-slate-600 text-xs">
          Every word is a weapon. Make them count.
        </div>
      </div>
    </div>
  );
}
