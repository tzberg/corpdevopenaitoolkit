# corpdevopenaitoolkit
OpenAI platform for corporate development teams

# CorpDev AI Platform v2.0

A streamlined AI-powered M&A analysis platform that delivers quick, actionable insights without the fluff. Built for corp dev teams who need speed and precision.

## What It Does

- **Target Sourcing**: Get 30 potential acquisition targets in seconds
- **Due Diligence**: Generate comprehensive DD frameworks instantly
- **Valuation Analysis**: Base/upside/downside DCF cases with comps
- **Market Analysis**: TAM/SAM/SOM + competitive landscape
- **Integration Planning**: Day 1 checklist through 100-day roadmap
- **Synergy Analysis**: Quantified revenue and cost synergies
- **Post-Analysis Chat**: Refine your analysis with follow-up questions

## Philosophy

This tool doesn't replace corp dev teams—it accelerates them. Every module is designed to give you quick numbers and frameworks so you can spend time on strategic thinking, not data gathering.

- **Minimal inputs**: Only what's necessary for each module
- **Quantitative focus**: Numbers over narrative
- **No fluff**: Straight to the insights
- **Fast iteration**: Chat with AI to refine analysis

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Your OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create account or sign in
3. Generate API key
4. Keep it secure—you'll paste it into the app settings

### 3. Launch
```bash
python main.py
```

Open browser to: **http://127.0.0.1:8080**

### 4. Configure
1. Click settings icon (⚙️) in top right
2. Paste your OpenAI API key
3. Save

## How To Use

### Quick Analysis Flow
1. **Select Module** → Choose your analysis type
2. **Fill Context** → Add only the required inputs (varies by module)
3. **Run Analysis** → Get results in 10-30 seconds
4. **Chat to Refine** → Ask follow-up questions to dig deeper

### Module-Specific Inputs

**Target Sourcing**
- Acquirer company name
- Target industry (e.g., "B2B SaaS")
- Geography (e.g., "North America")
- Revenue range (e.g., "$10M-$50M ARR")
- Strategic notes (optional)

**Due Diligence**
- Acquirer name
- Target name
- Strategic notes (optional)

**Valuation**
- Acquirer name
- Target name
- WACC % (optional, defaults to 9%)
- Terminal growth % (optional, defaults to 2.5%)
- Tax rate % (optional, defaults to 21%)
- Strategic notes (optional)

**Market Analysis**
- Acquirer name
- Target market (e.g., "Cloud Infrastructure")
- Sample company for comparison (optional)
- Strategic notes (optional)

**Integration Planning**
- Acquirer name
- Target name
- Strategic notes (optional)

**Synergy Analysis**
- Acquirer name
- Target name
- Strategic notes (optional)

## Cost Estimates

Using OpenAI API (gpt-4o-mini):
- Target sourcing: ~$0.05-0.10 per analysis
- Standard analysis: ~$0.02-0.05 per analysis
- Chat refinement: ~$0.01 per message
- Typical monthly usage (20 analyses): ~$1-2

Monitor usage: https://platform.openai.com/usage

## File Structure
```
├── main.py              # FastAPI backend
├── index.html           # React frontend (artifact)
├── requirements.txt     # Python dependencies
├── uploads/            # Uploaded files (auto-created)
└── README.md           # This file
```

## What Changed in v2.0

### Module-Specific Inputs
Each analysis now has tailored inputs. No more filling out irrelevant fields.

### Optimized Prompts
- Target sourcing delivers exactly 30 companies
- Valuation shows only cases and ranges
- All modules focus on numbers and frameworks

### Post-Analysis Chat
Refine your analysis by asking follow-up questions. Context is automatically included.

### Input Minimization
- Removed upload functionality from main flow (coming back as optional)
- Only show inputs relevant to selected module
- Inputs appear above module selection for better flow

## Development

Run with auto-reload:
```bash
uvicorn main:app --reload --port 8080
```

## API Endpoints

- `GET /health` - Health check
- `GET /api/status` - Module and dependency status
- `POST /api/analyze` - Run analysis
- `POST /api/chat` - Chat with AI assistant

## Troubleshooting

**"API Key Required" warning**
- Click settings gear icon
- Paste your OpenAI API key
- Click Save

**Analysis fails**
- Verify API key is correct
- Check OpenAI account has credits
- Try with simpler inputs first

**Missing dependencies**
```bash
pip install --upgrade -r requirements.txt
```

## Roadmap

- [ ] Save/load analysis history
- [ ] Export to PDF/PowerPoint
- [ ] Document upload for context
- [ ] Custom prompt templates
- [ ] Team collaboration features
- [ ] Pre-built industry templates

## Security

- API keys stored locally in browser (localStorage)
- No external data collection
- Files processed locally
- Only OpenAI API calls leave your machine

## License

MIT License - Use freely for commercial and personal projects

## Support

Questions? Issues? Found a bug?
- Check troubleshooting section above
- Review OpenAI documentation
- Verify all dependencies installed correctly

---

**Built for corp dev teams who value speed and precision over bloat.**
