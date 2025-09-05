import React, { useState, useEffect, useMemo } from 'react';
import {
  SafeAreaView,
  FlatList,
  StyleSheet,
  Text,
  View,
  StatusBar,
  TouchableOpacity,
  Alert,
} from 'react-native';

// --- 1. DATA TYPES (for TypeScript) ---
interface PlanItemData {
  id: string;
  icon: string;
  type: string;
  title: string;
  description: string;
}

interface DayPlanData {
  id: string;
  title: string;
  items: PlanItemData[];
}

// --- 2. RAW DATA & PARSING LOGIC ---

// In a real app, you would fetch this string from your backend API
const carePlanText = `
Day 1
🏃 Physical Activity: 30-minute brisk walk → Improves cardiovascular health, crucial for managing hypertension and reducing stroke risk.
🧘 Mental Wellness: 10-minute guided meditation for stress reduction → Helps lower blood pressure and manage anxiety related to health conditions.
🥗 Meals: Breakfast: Oatmeal with berries. Lunch: Grilled chicken salad. Dinner: Baked salmon with steamed vegetables → Low in sodium and saturated fats to support heart health and T2D management.
💧 Hydration: Drink 8 glasses of water throughout the day → Essential for kidney function and overall health.
❌ Avoid: Salty processed foods (canned soups, frozen meals) and sugary drinks → These can spike blood pressure and blood sugar levels.
✅ Today's risk reduction: Lowering sodium intake and engaging in light cardio directly addresses hypertension.
⚠ Consequences if skipped: Increased blood pressure, potential for blood sugar fluctuations, and higher strain on the heart.

Day 2
🏃 Physical Activity: 20 minutes of light stretching and mobility exercises → Improves circulation without over-exerting the heart.
🧘 Mental Wellness: Practice deep breathing exercises for 5 minutes, 3 times a day → Calms the nervous system and can lower heart rate.
🥗 Meals: Breakfast: Greek yogurt with nuts. Lunch: Lentil soup. Dinner: Turkey meatballs with whole wheat pasta → Balanced meals with fiber and lean protein for stable energy and blood sugar.
💧 Hydration: Infuse water with lemon or cucumber for flavor → Encourages consistent hydration.
❌ Avoid: Red meat and fried foods → High in saturated fats which can worsen cholesterol levels.
✅ Today's risk reduction: Focusing on lean proteins and fiber helps manage cholesterol and T2D.
⚠ Consequences if skipped: Poor cholesterol management, risk of digestive issues, and fatigue from unstable blood sugar.
`;

const EMOJI_MAP: Record<string, { type: string; title: string }> = {
  '🏃': { type: 'activity', title: 'Physical Activity' },
  '🧘': { type: 'wellness', title: 'Mental Wellness' },
  '🥗': { type: 'meals', title: 'Meals' },
  '💧': { type: 'hydration', title: 'Hydration' },
  '❌': { type: 'avoid', title: 'Avoid' },
  '✅': { type: 'risk_reduction', title: "Today's Risk Reduction" },
  '⚠': { type: 'consequences', title: 'Consequences if Skipped' },
};

const parseCarePlanText = (text: string): DayPlanData[] => {
  if (!text || typeof text !== 'string') {
    Alert.alert('Error', 'Invalid care plan data provided.');
    return [];
  }
  const dayBlocks = text.trim().split(/Day \d+/).filter(Boolean);
  return dayBlocks.map((block, index) => {
    const dayTitle = `Day ${index + 1}`;
    const lines = block.trim().split('\n');
    const items = lines
      .map((line) => {
        const icon = line.trim().charAt(0);
        const mappedInfo = EMOJI_MAP[icon];
        if (!mappedInfo) return null;
        const content = line.substring(1).trim();
        const [action, explanation] = content.split('→').map((s) => s.trim());
        return {
          id: `${dayTitle}-${mappedInfo.type}`,
          icon,
          type: mappedInfo.type,
          title: action || mappedInfo.title,
          description: explanation || '',
        };
      })
      .filter((item): item is PlanItemData => item !== null);
    return { id: dayTitle, title: dayTitle, items };
  });
};

// --- 3. REUSABLE UI COMPONENTS ---

const PlanItem: React.FC<{ item: PlanItemData }> = ({ item }) => (
  <View style={styles.itemContainer}>
    <Text style={styles.itemIcon}>{item.icon}</Text>
    <View style={styles.itemTextContainer}>
      <Text style={styles.itemTitle}>{item.title}</Text>
      <Text style={styles.itemDescription}>{item.description}</Text>
    </View>
  </View>
);

const DayPlanCard: React.FC<{ day: DayPlanData }> = ({ day }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const { standardItems, extraItems } = useMemo(() => {
    const standard = day.items.filter(
      (item) => item.type !== 'risk_reduction' && item.type !== 'consequences'
    );
    const extra = day.items.filter(
      (item) => item.type === 'risk_reduction' || item.type === 'consequences'
    );
    return { standardItems: standard, extraItems: extra };
  }, [day.items]);

  return (
    <View style={styles.card}>
      <Text style={styles.dayTitle}>{day.title}</Text>
      {standardItems.map((item) => (
        <PlanItem key={item.id} item={item} />
      ))}
      {extraItems.length > 0 && (
        <TouchableOpacity style={styles.button} onPress={() => setIsExpanded(!isExpanded)}>
          <Text style={styles.buttonText}>
            {isExpanded ? 'Show Less' : 'Show Risk & Consequences'}
          </Text>
        </TouchableOpacity>
      )}
      {isExpanded && extraItems.map((item) => <PlanItem key={item.id} item={item} />)}
    </View>
  );
};

// --- 4. MAIN SCREEN COMPONENT ---

export default function CarePlanScreen() {
  const [planData, setPlanData] = useState<DayPlanData[]>([]);

  useEffect(() => {
    const structuredData = parseCarePlanText(carePlanText);
    setPlanData(structuredData);
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <FlatList
        data={planData}
        renderItem={({ item }) => <DayPlanCard day={item} />}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.list}
        ListHeaderComponent={
          <View style={styles.header}>
            <Text style={styles.title}>Your 7-Day Care Plan</Text>
            <Text style={styles.subtitle}>Follow these daily recommendations to improve your health.</Text>
          </View>
        }
      />
    </SafeAreaView>
  );
}

// --- 5. STYLES ---

const styles = StyleSheet.create({
  // Main Screen Styles
  container: { flex: 1, backgroundColor: '#F8F9FA' },
  header: { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 10 },
  title: { fontSize: 28, fontWeight: 'bold', color: '#1D3557' },
  subtitle: { fontSize: 16, color: '#495057', marginTop: 8 },
  list: { paddingTop: 10, paddingBottom: 32 },

  // Day Card Styles
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    marginHorizontal: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 5,
  },
  dayTitle: { fontSize: 22, fontWeight: 'bold', marginBottom: 20, color: '#1A237E' },
  button: {
    backgroundColor: '#E8EAF6',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 10,
    marginBottom: 16,
  },
  buttonText: { color: '#3F51B5', fontWeight: 'bold', fontSize: 14 },

  // Plan Item Styles
  itemContainer: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 16 },
  itemIcon: { fontSize: 24, marginRight: 12, marginTop: 2, color: '#34495E' },
  itemTextContainer: { flex: 1 },
  itemTitle: { fontSize: 16, fontWeight: 'bold', color: '#2C3E50', marginBottom: 4 },
  itemDescription: { fontSize: 14, color: '#566573', lineHeight: 21 },
});

