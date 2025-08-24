# main.py
import os
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import pygame
import tempfile
import streamlit as st
from openai import OpenAI
import re
from dataclasses import dataclass, asdict
import threading
import queue
import time
import sys

if len(sys.argv) <= 1:
    sys.argv.append("--no-voice")

"""
ZOEY - Advanced Voice AI Storyteller
==================================
A sophisticated bedtime storytelling chatbot with:
- Voice recognition and synthesis
- Multi-language support
- Cultural personalization
- Persona memory system
- Multi-agent story generation pipeline

FUTURE ENHANCEMENTS (2-hour extension roadmap):
1. Advanced ML-based emotion detection from voice
2. Real-time story adaptation based on child engagement
3. Integration with smart home devices
4. Advanced RAG system with cultural story databases
5. Parent dashboard with story analytics
6. Multi-child family support with sibling story modes
"""

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class Persona:
    """User persona with all personalization data"""
    name: str
    age: int
    user_type: str  # 'child' or 'guardian'
    language: str
    location: str
    cultural_background: str
    interests: List[str]
    preferred_themes: List[str]
    story_length_preference: str
    voice_id: str  # Unique voice fingerprint
    linked_children: List[str] = None  # For guardians
    created_at: str = None
    last_interaction: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.linked_children is None:
            self.linked_children = []

class PersonaDatabase:
    """SQLite database for storing user personas"""
    
    def __init__(self, db_path: str = "personas.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                user_type TEXT NOT NULL,
                language TEXT NOT NULL,
                location TEXT NOT NULL,
                cultural_background TEXT NOT NULL,
                interests TEXT NOT NULL,
                preferred_themes TEXT NOT NULL,
                story_length_preference TEXT NOT NULL,
                voice_id TEXT UNIQUE NOT NULL,
                linked_children TEXT,
                created_at TEXT NOT NULL,
                last_interaction TEXT NOT NULL,
                UNIQUE(name, age, voice_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_persona(self, persona: Persona) -> bool:
        """Save or update a persona"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            persona.last_interaction = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO personas 
                (name, age, user_type, language, location, cultural_background, 
                 interests, preferred_themes, story_length_preference, voice_id, 
                 linked_children, created_at, last_interaction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                persona.name, persona.age, persona.user_type, persona.language,
                persona.location, persona.cultural_background,
                json.dumps(persona.interests), json.dumps(persona.preferred_themes),
                persona.story_length_preference, persona.voice_id,
                json.dumps(persona.linked_children), persona.created_at,
                persona.last_interaction
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error saving persona: {e}")
            return False
    
    def find_persona(self, voice_id: str = None, name: str = None, age: int = None) -> Optional[Persona]:
        """Find a persona by voice ID or name/age combination"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if voice_id:
                cursor.execute('SELECT * FROM personas WHERE voice_id = ?', (voice_id,))
            elif name and age:
                cursor.execute('SELECT * FROM personas WHERE name = ? AND age = ?', (name, age))
            else:
                return None
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return Persona(
                    name=row[1], age=row[2], user_type=row[3], language=row[4],
                    location=row[5], cultural_background=row[6],
                    interests=json.loads(row[7]), preferred_themes=json.loads(row[8]),
                    story_length_preference=row[9], voice_id=row[10],
                    linked_children=json.loads(row[11]) if row[11] else [],
                    created_at=row[12], last_interaction=row[13]
                )
            return None
        except Exception as e:
            st.error(f"Error finding persona: {e}")
            return None

class VoiceManager:
    """Handles voice recognition and synthesis with multi-language support"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.setup_tts()
        pygame.mixer.init()
    
    def setup_tts(self):
        """Configure text-to-speech engine"""
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Try to set a female voice for Zoey
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
        
        self.tts_engine.setProperty('rate', 150)  # Slower for children
        self.tts_engine.setProperty('volume', 0.9)
    
    def listen(self, timeout: int = 5) -> Optional[str]:
        """Listen for voice input and convert to text"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            
            st.info("Listening... Please speak!")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout)
            
            st.info("Processing your speech...")
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            st.warning("Sorry, I couldn't understand what you said. Please try again.")
            return None
        except Exception as e:
            st.error(f"Voice recognition error: {e}")
            return None
    
    def speak(self, text: str, language: str = 'en') -> None:
        """Convert text to speech with language support"""
        try:
            if language == 'en':
                # Use pyttsx3 for English (better quality)
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                # Use gTTS for other languages
                self.speak_multilingual(text, language)
        except Exception as e:
            st.error(f"Text-to-speech error: {e}")
    
    def speak_multilingual(self, text: str, language: str) -> None:
        """Speak text in different languages using gTTS"""
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts.save(fp.name)
                pygame.mixer.music.load(fp.name)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                os.unlink(fp.name)  # Clean up temp file
        except Exception as e:
            st.error(f"Multi-language TTS error: {e}")
    
    def generate_voice_id(self, audio_sample: str) -> str:
        """Generate a unique voice fingerprint (simplified)"""
        # In a real implementation, this would use voice biometrics
        # For now, we'll use a hash of the audio characteristics
        return hashlib.md5(audio_sample.encode()).hexdigest()[:12]

class CulturalContextEngine:
    """Engine for cultural story personalization"""
    
    def __init__(self):
        self.cultural_database = {
            'american': {
                'heroes': ['brave kids', 'superheroes', 'talking animals', 'space explorers'],
                'settings': ['neighborhoods', 'schools', 'parks', 'space stations', 'magical forests'],
                'values': ['friendship', 'honesty', 'helping others', 'courage', 'teamwork'],
                'characters': ['friendly dragons', 'wise owls', 'playful dolphins', 'brave mice'],
                'avoid': ['unfamiliar mythology', 'complex cultural concepts']
            },
            'indian': {
                'heroes': ['brave princes', 'wise animals', 'clever children', 'kind elephants'],
                'settings': ['magical villages', 'beautiful palaces', 'peaceful forests', 'holy rivers'],
                'values': ['respect for elders', 'courage', 'helping others', 'wisdom', 'kindness'],
                'characters': ['wise elephants', 'clever monkeys', 'brave tigers', 'kind peacocks'],
                'mythology': ['simple ramayana stories', 'panchatantra tales', 'akbar birbal stories']
            },
            'chinese': {
                'heroes': ['wise dragons', 'brave pandas', 'clever children', 'kind masters'],
                'settings': ['mountain temples', 'bamboo forests', 'ancient villages', 'magical gardens'],
                'values': ['harmony', 'wisdom', 'patience', 'respect', 'balance'],
                'characters': ['wise dragons', 'playful pandas', 'clever cranes', 'kind rabbits'],
                'festivals': ['chinese new year', 'lantern festival', 'moon festival']
            },
            'hispanic': {
                'heroes': ['brave children', 'wise abuelas', 'magical animals', 'kind families'],
                'settings': ['colorful villages', 'beautiful beaches', 'market squares', 'family homes'],
                'values': ['familia', 'generosity', 'joy', 'celebration', 'helping others'],
                'characters': ['wise parrots', 'playful iguanas', 'kind butterflies', 'brave jaguars'],
                'celebrations': ['dia de los muertos', 'quinceanera', 'posadas']
            }
        }
    
    def get_cultural_context(self, cultural_background: str) -> Dict:
        """Get cultural context for story generation"""
        return self.cultural_database.get(cultural_background.lower(), self.cultural_database['american'])
    
    def personalize_story_elements(self, persona: Persona) -> Dict:
        """Create personalized story elements based on persona"""
        cultural_context = self.get_cultural_context(persona.cultural_background)
        
        return {
            'protagonist_name': persona.name,
            'protagonist_age': persona.age,
            'setting_options': cultural_context['settings'],
            'character_options': cultural_context['characters'],
            'value_themes': cultural_context['values'],
            'cultural_elements': cultural_context,
            'interests': persona.interests,
            'language': persona.language,
            'story_length': persona.story_length_preference
        }

class StoryJudge:
    """AI judge to evaluate and improve story quality"""
    
    def __init__(self, client):
        self.client = client
    
    def evaluate_story(self, story: str, persona: Persona, theme: str) -> Dict:
        """Evaluate story quality and appropriateness"""
        
        evaluation_prompt = f"""
        You are an expert children's story evaluator. Evaluate this story for a {persona.age}-year-old child named {persona.name}.
        
        Story: {story}
        
        Evaluate on these criteria (rate 1-10 and provide brief feedback):
        1. Age Appropriateness (vocabulary, concepts, emotional complexity)
        2. Cultural Sensitivity (appropriate for {persona.cultural_background} background)
        3. Engagement Level (exciting, interesting for the child)
        4. Moral/Educational Value (clear positive message)
        5. Story Structure (clear beginning, middle, end)
        6. Length Appropriateness (suitable for {persona.story_length_preference})
        7. Personalization (uses child's name effectively, relates to their interests: {persona.interests})
        
        Respond in JSON format:
        {{
            "overall_score": 0-10,
            "age_appropriateness": {{"score": 0-10, "feedback": "brief comment"}},
            "cultural_sensitivity": {{"score": 0-10, "feedback": "brief comment"}},
            "engagement": {{"score": 0-10, "feedback": "brief comment"}},
            "moral_value": {{"score": 0-10, "feedback": "brief comment"}},
            "story_structure": {{"score": 0-10, "feedback": "brief comment"}},
            "length": {{"score": 0-10, "feedback": "brief comment"}},
            "personalization": {{"score": 0-10, "feedback": "brief comment"}},
            "improvement_suggestions": ["suggestion1", "suggestion2", "suggestion3"],
            "needs_revision": true/false
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": evaluation_prompt}],
                max_tokens=1000,
                temperature=0.2
            )
            
            # Extract JSON from response
            evaluation_text = response.choices[0].message.content
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
                return evaluation
            else:
                return {"overall_score": 5, "needs_revision": False, "improvement_suggestions": []}
                
        except Exception as e:
            st.error(f"Story evaluation error: {e}")
            return {"overall_score": 5, "needs_revision": False, "improvement_suggestions": []}

class StoryGenerator:
    """Main story generation engine with multi-agent pipeline"""
    
    def __init__(self, client):
        self.client = client
        self.cultural_engine = CulturalContextEngine()
        self.judge = StoryJudge(client)
    
    def generate_story(self, persona: Persona, theme: str, max_iterations: int = 3) -> Tuple[str, Dict]:
        """Generate a personalized story with iterative improvement"""
        
        # Get personalized story elements
        story_elements = self.cultural_engine.personalize_story_elements(persona)
        
        story = None
        evaluation = None
        
        for iteration in range(max_iterations):
            if iteration == 0:
                # Initial story generation
                story = self._generate_initial_story(persona, theme, story_elements)
            else:
                # Story improvement based on judge feedback
                story = self._improve_story(story, evaluation, persona, theme, story_elements)
            
            # Evaluate the story
            evaluation = self.judge.evaluate_story(story, persona, theme)
            
            # If story is good enough, break
            if not evaluation.get('needs_revision', False) or evaluation.get('overall_score', 0) >= 8:
                break
            
            st.info(f"Improving story... (Iteration {iteration + 1})")
        
        return story, evaluation
    
    def _generate_initial_story(self, persona: Persona, theme: str, story_elements: Dict) -> str:
        """Generate the initial story"""
        
        # Determine story length based on preference
        length_guide = {
            'short': '300-500 words (3-5 minutes reading)',
            'medium': '500-800 words (5-8 minutes reading)', 
            'long': '800-1200 words (8-12 minutes reading)'
        }
        
        story_prompt = f"""
        You are Zoey, a magical storyteller who creates personalized bedtime stories for children.
        
        Create a {persona.story_length_preference} bedtime story ({length_guide.get(persona.story_length_preference, '500-800 words')}) for {persona.name}, age {persona.age}.
        
        PERSONALIZATION REQUIREMENTS:
        - Main character: {persona.name} (age {persona.age})
        - Cultural background: {persona.cultural_background}
        - Child's interests: {', '.join(persona.interests)}
        - Language: {persona.language} (but write in English - will be translated if needed)
        - Theme: {theme}
        
        STORY ELEMENTS TO INCLUDE:
        - Setting: Choose from {story_elements['setting_options']}
        - Supporting characters: Choose from {story_elements['character_options']}
        - Values to highlight: {story_elements['value_themes']}
        - Make {persona.name} the brave hero/heroine of the story
        
        REQUIREMENTS:
        - Age-appropriate vocabulary for {persona.age}-year-old
        - Clear moral/lesson that relates to {theme}
        - Engaging and exciting but suitable for bedtime
        - Use {persona.name} throughout the story as the main character
        - Include familiar cultural elements appropriate for {persona.cultural_background} background
        - Happy, comforting ending perfect for sleep
        
        Write a complete story with clear beginning, middle, and end. Make it magical and wonderful!
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": story_prompt}],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"Story generation error: {e}")
            return f"Once upon a time, there was a wonderful child named {persona.name} who went on a magical adventure..."
    
    def _improve_story(self, story: str, evaluation: Dict, persona: Persona, theme: str, story_elements: Dict) -> str:
        """Improve the story based on judge feedback"""
        
        suggestions = evaluation.get('improvement_suggestions', [])
        
        improvement_prompt = f"""
        You are Zoey, a magical storyteller. Please improve this bedtime story based on the feedback provided.
        
        ORIGINAL STORY:
        {story}
        
        IMPROVEMENT SUGGESTIONS:
        {'. '.join(suggestions)}
        
        REQUIREMENTS:
        - Keep {persona.name} as the main character
        - Maintain age-appropriateness for {persona.age}-year-old
        - Keep the {persona.cultural_background} cultural context
        - Address the specific suggestions while keeping the story engaging
        - Ensure proper length for {persona.story_length_preference} story
        
        Rewrite the improved story:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": improvement_prompt}],
                max_tokens=1500,
                temperature=0.6
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"Story improvement error: {e}")
            return story  # Return original if improvement fails

class PersonaBuilder:
    """Interactive persona building system"""
    
    def __init__(self, voice_manager: VoiceManager):
        self.voice_manager = voice_manager
    
    def detect_user_type(self, initial_input: str) -> str:
        """Detect if user is a child or guardian"""
        child_indicators = ['me', 'i want', 'my favorite', 'i like', 'can you tell me']
        parent_indicators = ['my child', 'for my daughter', 'for my son', 'bedtime story for', 'my kid']
        
        input_lower = initial_input.lower()
        
        child_score = sum(1 for indicator in child_indicators if indicator in input_lower)
        parent_score = sum(1 for indicator in parent_indicators if indicator in input_lower)
        
        return 'child' if child_score >= parent_score else 'guardian'
    
    def build_persona_interactive(self, user_type: str, use_voice: bool = False) -> Persona:
        """Build persona through interactive conversation"""
        if user_type == 'child':
            return self._build_child_persona(use_voice)
        else:
            return self._build_guardian_persona(use_voice)
    
    def _get_input(self, prompt: str, use_voice: bool) -> str:
        """Get input via voice or text"""
        if use_voice:
            st.info(prompt)
            self.voice_manager.speak(prompt)
            response = self.voice_manager.listen()
            if response:
                st.success(f"You said: {response}")
                return response
            else:
                # Fallback to text input
                return st.text_input("Please type your answer:", key=f"input_{len(prompt)}")
        else:
            return st.text_input(prompt, key=f"input_{len(prompt)}")
    
    def _build_child_persona(self, use_voice: bool) -> Persona:
        """Build persona by talking directly to the child"""
        st.subheader("Hi there! Let me get to know you better!")
        
        # Get basic information
        name = self._get_input("What's your name, little friend?", use_voice)
        if not name:
            return None
            
        age_str = self._get_input(f"How old are you, {name}?", use_voice)
        try:
            age = int(re.findall(r'\d+', age_str)[0]) if age_str else 6
        except:
            age = 6
        
        # Get interests
        interests_str = self._get_input(f"What are your favorite things, {name}? Like animals, superheroes, princesses?", use_voice)
        interests = [i.strip() for i in interests_str.split(',') if i.strip()] if interests_str else ['animals', 'adventures']
        
        # Get location/cultural info (simplified for child)
        location = self._get_input("Where do you live? What city or country?", use_voice) or "Unknown"
        
        # Infer cultural background from location or name
        cultural_background = self._infer_cultural_background(name, location)
        
        # Story preferences
        length_pref = st.selectbox(
            "How long should your stories be?",
            ["short", "medium", "long"],
            format_func=lambda x: {"short": "Short (3-5 minutes)", "medium": "Medium (5-8 minutes)", "long": "Long (8-12 minutes)"}[x]
        )
        
        # Language preference
        language = st.selectbox("What language do you prefer?", ["en", "es", "hi", "zh", "fr"], 
                               format_func=lambda x: {"en": "English", "es": "Spanish", "hi": "Hindi", "zh": "Chinese", "fr": "French"}[x])
        
        # Generate voice ID (simplified)
        voice_id = hashlib.md5(f"{name}_{age}_{datetime.now()}".encode()).hexdigest()[:12]
        
        return Persona(
            name=name,
            age=age,
            user_type='child',
            language=language,
            location=location,
            cultural_background=cultural_background,
            interests=interests,
            preferred_themes=['friendship', 'adventure', 'helping others'],
            story_length_preference=length_pref,
            voice_id=voice_id
        )
    
    def _build_guardian_persona(self, use_voice: bool) -> Persona:
        """Build persona with guardian providing child's information"""
        st.subheader("Hello! Please tell me about your child")
        
        child_name = self._get_input("What's your child's name?", use_voice)
        if not child_name:
            return None
            
        age_str = self._get_input(f"How old is {child_name}?", use_voice)
        try:
            age = int(re.findall(r'\d+', age_str)[0]) if age_str else 6
        except:
            age = 6
        
        # More detailed questions for guardians
        interests_str = self._get_input(f"What are {child_name}'s favorite things? (animals, sports, books, etc.)", use_voice)
        interests = [i.strip() for i in interests_str.split(',') if i.strip()] if interests_str else ['animals', 'adventures']
        
        location = self._get_input("Where are you located? (City, Country)", use_voice) or "Unknown"
        
        # Cultural background
        cultural_options = ['American', 'Indian', 'Chinese', 'Hispanic', 'African', 'European', 'Other']
        cultural_background = st.selectbox("What's your cultural background?", cultural_options).lower()
        
        # Preferred themes
        theme_options = ['friendship', 'courage', 'kindness', 'honesty', 'family values', 'helping others', 'adventure']
        preferred_themes = st.multiselect("What values/themes would you like in the stories?", theme_options)
        
        # Story length
        length_pref = st.selectbox(
            "Preferred story length:",
            ["short", "medium", "long"],
            format_func=lambda x: {"short": "Short (3-5 minutes)", "medium": "Medium (5-8 minutes)", "long": "Long (8-12 minutes)"}[x]
        )
        
        # Language
        language = st.selectbox("Preferred language:", ["en", "es", "hi", "zh", "fr"], 
                               format_func=lambda x: {"en": "English", "es": "Spanish", "hi": "Hindi", "zh": "Chinese", "fr": "French"}[x])
        
        voice_id = hashlib.md5(f"{child_name}_{age}_{datetime.now()}".encode()).hexdigest()[:12]
        
        return Persona(
            name=child_name,
            age=age,
            user_type='child',  # The persona is for the child
            language=language,
            location=location,
            cultural_background=cultural_background,
            interests=interests,
            preferred_themes=preferred_themes or ['friendship', 'adventure'],
            story_length_preference=length_pref,
            voice_id=voice_id
        )
    
    def _infer_cultural_background(self, name: str, location: str) -> str:
        """Simple cultural background inference"""
        name_lower = name.lower()
        location_lower = location.lower()
        
        # Indian names
        if any(indicator in name_lower for indicator in ['raj', 'priya', 'arjun', 'lakshmi', 'krishna', 'arun', 'maya']):
            return 'indian'
        
        # Chinese names
        if any(indicator in name_lower for indicator in ['li', 'wang', 'chen', 'zhang', 'ming', 'wei']):
            return 'chinese'
        
        # Hispanic names
        if any(indicator in name_lower for indicator in ['garcia', 'rodriguez', 'martinez', 'jose', 'maria', 'carlos']):
            return 'hispanic'
        
        # Location-based inference
        if any(country in location_lower for country in ['india', 'china', 'mexico', 'spain']):
            return location_lower.split()[0]
        
        return 'american'  # Default

class ZoeyStorytellerApp:
    """Main application class"""
    def __init__(self):
        self.db = PersonaDatabase()
        self.voice_manager = VoiceManager()
        self.persona_builder = PersonaBuilder(self.voice_manager)
        self.story_generator = StoryGenerator(client)
        
        # Initialize session state
        if 'current_persona' not in st.session_state:
            st.session_state.current_persona = None
        if 'story_history' not in st.session_state:
            st.session_state.story_history = []

    def introduce_zoey(self):
        """Introduce Zoey to the user"""
        if "--no-voice" not in sys.argv:
            intro_text = """
            Hello! I'm Zoey, your magical AI storyteller! 
            I create personalized bedtime stories just for you. 
            Every story is special because YOU are the hero!
            
            I can tell stories in different languages, include your favorite things,
            and make sure every adventure ends with sweet dreams.
            
            Let's create some magic together!
            """
            self.voice_manager.speak(intro_text)
        
        st.success("Hi! I'm Zoey, your magical storyteller!")
        st.info("""
        **What makes me special:**
        - I create stories where YOU are the hero
        - Stories adapt to your age, interests, and culture
        - Multi-language support
        - Voice interaction available
        - Every story is unique and personalized
        
        Ready to begin our magical journey?
        """)
    
    def run(self):
        """Run the main application"""
        st.set_page_config(
            page_title="Zoey - AI Storyteller",
            page_icon="🌙",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better UI
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .story-box {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 2rem;
            border-radius: 15px;
            border-left: 5px solid #ff6b6b;
            margin: 1rem 0;
        }
        .persona-info {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🌙 Zoey - Your Magical AI Storyteller</h1>
            <p>Personalized bedtime stories that make every child the hero of their own adventure!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar for controls
        with st.sidebar:
            st.title("🎭 Story Controls")
            
            use_voice = st.checkbox("🎤 Use Voice Mode", value=False)
            
            if st.button("👋 Meet Zoey"):
                self.introduce_zoey()
            
            st.divider()
            
            # Persona management
            st.subheader("👤 User Profile")
            
            if st.session_state.current_persona:
                st.markdown(f"""
                <div class="persona-info">
                    <h4>Current User: {st.session_state.current_persona.name}</h4>
                    <p>Age: {st.session_state.current_persona.age}</p>
                    <p>Interests: {', '.join(st.session_state.current_persona.interests)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔄 Switch User"):
                    st.session_state.current_persona = None
                    st.rerun()
            else:
                if st.button("✨ Create New Profile"):
                    self.setup_new_persona(use_voice)
        
        # Main content area
        if not st.session_state.current_persona:
            self.show_welcome_screen()
        else:
            self.show_story_interface(use_voice)

    def setup_new_persona(self, use_voice):
        """Setup a new persona"""
        st.subheader("✨ Let's create your magical profile!")
        
        # Detect user type
        initial_input = st.text_input("Tell me, who is this for? (e.g., 'for my 6-year-old daughter' or 'I want stories for me')")
        
        if initial_input:
            user_type = self.persona_builder.detect_user_type(initial_input)
            persona = self.persona_builder.build_persona_interactive(user_type, use_voice)
            
            if persona:
                # Save persona
                if self.db.save_persona(persona):
                    st.session_state.current_persona = persona
                    st.success(f"Profile created for {persona.name}! 🎉")
                    st.rerun()
                else:
                    st.error("Failed to save profile. Please try again.")

    def show_welcome_screen(self):
        """Show welcome screen for new users"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ## 🌟 Welcome to Zoey's Magical World!
            
            I'm here to create amazing bedtime stories just for you!
            Every story is special because **you** are the hero.
            
            ### ✨ What I can do:
            - Create personalized stories with you as the main character
            - Adapt stories to your age, interests, and cultural background  
            - Tell stories in multiple languages
            - Use voice interaction for a magical experience
            - Remember your preferences for future stories
            
            ### 🚀 Ready to start?
            Click "Create New Profile" in the sidebar to begin our adventure!
            """)

    def show_story_interface(self, use_voice):
        """Show the main story interface"""
        persona = st.session_state.current_persona
        
        st.subheader(f"🌟 Hello {persona.name}! Ready for a story?")
        
        # Story theme selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Choose your adventure theme:**")
            theme_options = [
                "Friendship and Kindness",
                "Brave Adventures", 
                "Magical Creatures",
                "Space Exploration",
                "Ocean Adventures",
                "Forest Magic",
                "Helping Others",
                "Overcoming Fears",
                "Family Love",
                "Custom Theme"
            ]
            
            selected_theme = st.selectbox("Theme:", theme_options)
            
            if selected_theme == "Custom Theme":
                selected_theme = st.text_input("What theme would you like?")
        
        with col2:
            st.write("**Story preferences:**")
            st.write(f"Length: {persona.story_length_preference}")
            st.write(f"Language: {persona.language}")
            st.write(f"Age: {persona.age} years old")
        
        # Generate story button
        if st.button("✨ Create My Story!", type="primary"):
            if selected_theme:
                self.generate_and_display_story(persona, selected_theme, use_voice)
            else:
                st.warning("Please select a theme first!")
        
        # Story history
        if st.session_state.story_history:
            st.divider()
            st.subheader("📚 Your Story Collection")
            
            for i, (story_theme, story_text, story_eval) in enumerate(reversed(st.session_state.story_history[-5:])):
                with st.expander(f"Story {len(st.session_state.story_history) - i}: {story_theme}"):
                    st.markdown(f"""
                    <div class="story-box">
                        {story_text.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if story_eval:
                        st.caption(f"Story Quality Score: {story_eval.get('overall_score', 'N/A')}/10")

    def generate_and_display_story(self, persona, theme, use_voice):
        """Generate and display a new story"""
        with st.spinner("✨ Creating your magical story..."):
            try:
                story, evaluation = self.story_generator.generate_story(persona, theme)
                
                # Display the story
                st.markdown(f"""
                <div class="story-box">
                    <h3>🌟 {theme} Adventure</h3>
                    {story.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
                
                # Show story quality
                if evaluation:
                    score = evaluation.get('overall_score', 0)
                    if score >= 8:
                        st.success(f"📖 Story Quality: Excellent ({score}/10)")
                    elif score >= 6:
                        st.info(f"📖 Story Quality: Good ({score}/10)")
                    else:
                        st.warning(f"📖 Story Quality: Fair ({score}/10)")
                
                # Voice narration
                if use_voice and "--no-voice" not in sys.argv:
                    if st.button("🔊 Listen to your story"):
                        with st.spinner("🎭 Zoey is telling your story..."):
                            self.voice_manager.speak(story, persona.language)
                
                # Save to history
                st.session_state.story_history.append((theme, story, evaluation))
                
                # Update persona last interaction
                persona.last_interaction = datetime.now().isoformat()
                self.db.save_persona(persona)
                
            except Exception as e:
                st.error(f"Sorry, I had trouble creating your story. Error: {e}")
                st.info("💡 Make sure you have set your OpenAI API key as an environment variable.")


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        st.error("🔑 Please set your OPENAI_API_KEY environment variable!")
        st.info("You can get an API key from https://platform.openai.com/api-keys")
        st.stop()
    
    app = ZoeyStorytellerApp()
    app.run()