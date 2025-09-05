import React, { useState, useEffect } from 'react';
import { SafeAreaView, StyleSheet, Text, View, StatusBar, TouchableOpacity, Alert, ScrollView, ActivityIndicator } from 'react-native';
import { post, get } from '../../src/lib/api';

export default function CarePlanScreen() {
  const [loading, setLoading] = useState(false);
  const [todayContent, setTodayContent] = useState<string | null>(null);
  const [dayNumber, setDayNumber] = useState<number | null>(null);
  const [isCompleted, setIsCompleted] = useState<boolean>(false);
  const [completedChoice, setCompletedChoice] = useState<'yes' | 'no' | null>(null);

  const fetchToday = async () => {
    setLoading(true);
    try {
      const { ok, data } = await get('/care_plan/today', true);
      if (!ok) throw new Error((data as any).error || 'Fetch failed');
      const d: any = data;
      setTodayContent(d.content);
      setDayNumber(d.day_number);
      setIsCompleted(!!d.is_completed);
      setCompletedChoice((d.completed_choice as any) || null);
    } catch (e: any) {
      setTodayContent(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchToday();
  }, []);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const { ok, data } = await post('/care_plan/generate', {}, true);
      if (!ok) throw new Error((data as any).error || 'Generate failed');
      await fetchToday();
    } catch (e: any) {
      Alert.alert('Error', e.message || 'Could not generate plan');
      setLoading(false);
    }
  };

  const handleComplete = async (choice: 'yes' | 'no') => {
    setLoading(true);
    try {
      const { ok, data } = await post('/care_plan/complete', { choice }, true);
      if (!ok) throw new Error((data as any).error || 'Update failed');
      setIsCompleted(true);
      setCompletedChoice(choice);
    } catch (e: any) {
      Alert.alert('Error', e.message || 'Could not update status');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <View style={styles.header}>
        <Text style={styles.title}>Your Care Plan</Text>
        <Text style={styles.subtitle}>Only today's unlocked day will be shown.</Text>
      </View>

      {loading && (
        <View style={{ padding: 16 }}>
          <ActivityIndicator />
        </View>
      )}

      {!todayContent && !loading && (
        <View style={{ padding: 16 }}>
          <Text style={{ marginBottom: 12 }}>No plan available yet.</Text>
          <TouchableOpacity style={styles.primaryButton} onPress={handleGenerate}>
            <Text style={styles.primaryButtonText}>Generate Care Plan</Text>
          </TouchableOpacity>
        </View>
      )}

      {todayContent && (
        <ScrollView contentContainerStyle={styles.list}>
          <View style={styles.card}>
            <Text style={styles.dayTitle}>Day {dayNumber}</Text>
            <Text style={styles.planText}>{todayContent}</Text>
            {!isCompleted ? (
              <View style={{ marginTop: 16 }}>
                <Text style={{ marginBottom: 8 }}>Did you follow today's care plan?</Text>
                <View style={{ flexDirection: 'row' }}>
                  <TouchableOpacity style={styles.choiceYes} onPress={() => handleComplete('yes')}>
                    <Text style={styles.choiceText}>Yes</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.choiceNo} onPress={() => handleComplete('no')}>
                    <Text style={styles.choiceText}>No</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ) : (
              <Text style={{ marginTop: 12 }}>Completed: {completedChoice}</Text>
            )}
          </View>
        </ScrollView>
      )}
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
  dayTitle: { fontSize: 22, fontWeight: 'bold', marginBottom: 16, color: '#1A237E' },
  planText: { fontSize: 15, color: '#2C3E50', lineHeight: 22 },
  primaryButton: { backgroundColor: '#0a7ea4', padding: 14, borderRadius: 10, alignItems: 'center' },
  primaryButtonText: { color: '#fff', fontWeight: 'bold' },
  choiceYes: { backgroundColor: '#2ECC71', padding: 12, borderRadius: 8, marginRight: 10 },
  choiceNo: { backgroundColor: '#E74C3C', padding: 12, borderRadius: 8 },
  choiceText: { color: '#fff', fontWeight: '600' },
  button: {
    backgroundColor: '#E8EAF6',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 10,
    marginBottom: 16,
  },
  buttonText: { color: '#3F51B5', fontWeight: 'bold', fontSize: 14 },

  // Removed old per-item UI; using raw formatted plan text per backend
});

