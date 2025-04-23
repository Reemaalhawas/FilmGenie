import React, { useState, useCallback, useRef } from 'react';
import Particles from '@tsparticles/react';
import { loadSlim } from '@tsparticles/slim';
import { Element } from 'react-scroll';
import { motion, AnimatePresence } from 'framer-motion';

const faqItems = [
  {
    question: 'Is Film Genie free to use?',
    answer: 'Yes! Film Genie is completely free for all users.',
  },
  {
    question: 'Do I need to create an account?',
    answer: 'Nope! Just tap the lamp, take the quiz, and enjoy your results.',
  },
  {
    question: 'How accurate are the recommendations?',
    answer: 'We use your mood and preferences to show you high-match films — it’s like magic!',
  },
];

const LandingPage = () => {
  const [isRubbing, setIsRubbing] = useState(false);
  const [isTeleporting, setIsTeleporting] = useState(false);
  const [lampCoords, setLampCoords] = useState({ x: '50%', y: '50%' });
  const lampRef = useRef(null);
  const [faqOpenIndex, setFaqOpenIndex] = useState(null);

  const particlesInit = useCallback(async (engine) => {
    await loadSlim(engine);
  }, []);

  const handleLampClick = () => {
    const audio = new Audio('/magic.mp3');
    audio.play();

    if (lampRef.current) {
      const rect = lampRef.current.getBoundingClientRect();
      setLampCoords({
        x: `${rect.left + rect.width / 2}px`,
        y: `${rect.top + rect.height / 2}px`,
      });
    }

    setIsTeleporting(true);
    setTimeout(() => {
      window.location.href = "/quiz";
    }, 2000);
  };

  const particlesOptions = {
    particles: {
      number: { value: 80, density: { enable: true, value_area: 300 } },
      color: { value: "#FFD700" },
      shape: { type: "star" },
      opacity: { value: 0.9, random: true },
      size: { value: 3, random: true },
      move: {
        enable: true,
        speed: 2,
        direction: "none",
        random: true,
        outModes: { default: "out" },
      },
    },
    detectRetina: true,
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#1A1A40] to-black text-white font-poppins relative overflow-x-hidden">
      <div className="absolute inset-0 pointer-events-none stars"></div>

      <header className="flex justify-between items-center p-4">
      </header>

      <Element name="lamp">
        <div className="flex flex-col items-center justify-center text-center pt-12 pb-24 px-4">
          <h1 className="text-5xl font-bold mb-6 text-gold">Your Movie Wish, Granted.</h1>
          <p className="text-xl text-gray-300 mb-8 max-w-xl">Tap the lamp. Find your film.</p>
          <div
            ref={lampRef}
            className="relative mb-10 cursor-pointer"
            onMouseEnter={() => setIsRubbing(true)}
            onMouseLeave={() => setIsRubbing(false)}
            onClick={handleLampClick}
          >
            <div className={`lamp-icon text-[8rem] ${isRubbing ? 'rub' : ''}`}>🪔</div>
            {isRubbing && (
              <Particles id="tsparticles" init={particlesInit} options={particlesOptions} className="absolute inset-0 z-[-1]" />
            )}
          </div>
        </div>
      </Element>

      {isTeleporting && (
        <div className="fixed inset-0 z-50 pointer-events-none">
          <div className="absolute inset-0 bg-black bg-opacity-70 animate-fadeIn"></div>
          <Particles
            id="magic-transition"
            init={particlesInit}
            options={{
              ...particlesOptions,
              particles: {
                ...particlesOptions.particles,
                number: { value: 200 },
                size: { value: 5, random: true },
                move: { ...particlesOptions.particles.move, speed: 4 },
              },
            }}
            className="absolute inset-0"
          />
          <div className="absolute puff" style={{
            left: lampCoords.x,
            top: lampCoords.y,
            transform: 'translate(-50%, -50%)',
          }} />
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-[#FFD700] font-bold text-xl">
            The Genie is preparing your wish...
          </div>
        </div>
      )}

      <Element name="how-it-works">
        <section className="py-16 px-6 max-w-6xl mx-auto text-center">
          <h2 className="text-3xl text-gold font-bold mb-10">How It Works</h2>
          <div className="grid gap-6 md:grid-cols-3">
            {[
              ['Take the Quiz', 'Tell us your mood and preferences.'],
              ['Swipe Through Magic', 'Choose what sparks your vibe.'],
              ['Get Genie Picks', 'Your perfect film, no more endless scrolling.']
            ].map(([title, desc], i) => (
              <motion.div
                key={i}
                whileHover={{ scale: 1.05 }}
                className="bg-white/5 border border-[#585D9C] p-6 rounded-xl shadow"
              >
                <h3 className="text-xl text-gold font-semibold mb-2">{title}</h3>
                <p className="text-gray-300 text-sm">{desc}</p>
              </motion.div>
            ))}
          </div>
        </section>
      </Element>

      {/* FAQ */}
      <Element name="faq">
        <section className="py-16 px-6 max-w-3xl mx-auto">
          <h2 className="text-3xl text-gold font-bold mb-8 text-center">Frequently Asked Questions</h2>
          {faqItems.map((item, i) => (
            <div key={i} className="mb-4 border border-[#585D9C] rounded-lg">
              <button
                onClick={() => setFaqOpenIndex(faqOpenIndex === i ? null : i)}
                className="w-full px-4 py-3 text-left text-white font-medium bg-[#1A1A40] hover:bg-[#252552]"
              >
                {item.question}
              </button>
              <AnimatePresence>
                {faqOpenIndex === i && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="px-4 py-3 text-sm bg-black/30 text-gray-300"
                  >
                    {item.answer}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ))}
        </section>
      </Element>

      {/* Footer */}
      <footer className="p-4 text-center text-sm text-gray-500 border-t border-[#333]">
        © 2025 Film Genie. All rights reserved.
      </footer>

      {/* Styles */}
      <style jsx>{`
        .stars {
          background: url('https://www.transparenttextures.com/patterns/stardust.png');
          opacity: 0.5;
          height: 100%;
          width: 100%;
          animation: twinkle 10s infinite;
        }
        @keyframes twinkle {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 0.8; }
        }
        .font-poppins {
          font-family: 'Poppins', sans-serif;
        }
        .text-gold { color: #FFD700; }
        .lamp-icon { transition: transform 0.3s ease; }
        .rub { animation: shake 0.5s infinite, glow 1.5s infinite; }
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-5px); }
          75% { transform: translateX(5px); }
        }
        @keyframes glow {
          0%, 100% { text-shadow: 0 0 10px rgba(255, 215, 0, 0.4); }
          50% { text-shadow: 0 0 20px rgba(255, 215, 0, 1); }
        }
        .puff {
          width: 300px;
          height: 300px;
          background: radial-gradient(circle, rgba(255,215,0,0.6) 0%, transparent 70%);
          border-radius: 50%;
          animation: puff-glow 1s ease-out forwards;
        }
        @keyframes puff-glow {
          0% { transform: scale(0.3); opacity: 0; }
          50% { transform: scale(1.2); opacity: 1; }
          100% { transform: scale(1.8); opacity: 0; }
        }
      `}</style>
    </div>
  );
};

export default LandingPage;
