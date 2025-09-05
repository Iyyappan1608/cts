import { Ionicons } from '@expo/vector-icons';
import { Link, useNavigation, useRouter } from 'expo-router';
import React, { useLayoutEffect } from 'react';
import { 
  ActivityIndicator, 
  FlatList, 
  SafeAreaView, 
  ScrollView, 
  StatusBar, 
  StyleSheet, 
  Text, 
  TouchableOpacity, 
  View 
} from 'react-native';
import AnimatedCard from '../../components/AnimatedCard';
import DashboardCard from '../../components/DashboardCard';
import { Colors } from '../../constants/Colors';
import { useData, VitalReading } from '../../src/context/DataContext';

type CarePlanItem = {
    id: string;
    task: string;
    completed: boolean;
    icon: 'walk-outline' | 'medkit-outline' | 'restaurant-outline';
};

export default function HomeScreen() {
  const navigation = useNavigation();
  const router = useRouter();
  const { userData, isLoading, toggleCarePlanItem } = useData();

  useLayoutEffect(() => {
    navigation.setOptions({
      headerRight: () => (
        <TouchableOpacity 
          onPress={() => router.push('/live-monitor')} 
          style={{ marginRight: 15 }}
        >
          <Ionicons name="watch-outline" size={28} color={Colors.primary} />
        </TouchableOpacity>
      ),
      headerStyle: { backgroundColor: Colors.background, },
      headerShadowVisible: false,
    });
  }, [navigation, router]);

  if (isLoading || !userData) {
    return (
      <SafeAreaView style={[styles.container, styles.centerContent]}>
        <ActivityIndicator size="large" color={Colors.primary} />
      </SafeAreaView>
    );
  }
  
  const latestVitals = userData.vitalHistory.length > 0 ? userData.vitalHistory[0] : null;

  // --- THIS LOGIC IS NOW CORRECTED ---
  const hasDiabetes = userData.predictedConditions?.some((condition: string) => condition.toLowerCase().includes('diabetes'));
  const hasHypertension = userData.predictedConditions?.some((condition: string) => condition.toLowerCase().includes('hypertension'));
  
  const renderHistoryItem = ({ item }: { item: VitalReading }) => (
    <View style={styles.historyItem}>
      <View style={styles.historyHeader}>
        <Ionicons name="calendar-outline" size={20} color={Colors.primary} />
        <Text style={styles.historyDate}>{new Date(item.date).toLocaleString()}</Text>
      </View>
      <View style={styles.historyVitals}>
        {item.glucose && <Text style={styles.historyText}>Glucose: {item.glucose} mg/dL</Text>}
        {item.bloodPressure && <Text style={styles.historyText}>BP: {item.bloodPressure} mmHg</Text>}
        {item.heartRate && <Text style={styles.historyText}>HR: {item.heartRate} bpm</Text>}
        {item.notes && <Text style={styles.historyNotes}>Notes: {item.notes}</Text>}
      </View>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.contentContainer}>
        <StatusBar barStyle="dark-content" />
        <Text style={styles.headerTitle}>Good Morning, {userData.name}</Text>
        <Text style={styles.headerSubtitle}>Here is your health summary.</Text>

        <AnimatedCard index={0}>
          <Link href="/add-entry" asChild>
            <TouchableOpacity style={styles.addButton}>
              <Ionicons name="search-outline" size={24} color="#FFFFFF" />
              <Text style={styles.addButtonText}>Check for Chronic Diseases</Text>
            </TouchableOpacity>
          </Link>
        </AnimatedCard>

        {(hasDiabetes || hasHypertension) && (
          <AnimatedCard index={1}>
            <DashboardCard icon="stats-chart-outline" title="Drill-down Analysis">
              {hasDiabetes && (
                <Link href="/diabetes-check" asChild>
                  <TouchableOpacity style={styles.drilldownButton}>
                    <Text style={styles.drilldownButtonText}>Analyze Diabetes Subtype</Text>
                    <Ionicons name="chevron-forward-outline" size={20} color="#FFFFFF" />
                  </TouchableOpacity>
                </Link>
              )}
              {hasHypertension && (
                <Link href="/hypertension-check" asChild>
                  <TouchableOpacity style={[styles.drilldownButton, {marginTop: hasDiabetes ? 10 : 0}]}>
                    <Text style={styles.drilldownButtonText}>Analyze Hypertension Stage</Text>
                    <Ionicons name="chevron-forward-outline" size={20} color="#FFFFFF" />
                  </TouchableOpacity>
                </Link>
              )}
            </DashboardCard>
          </AnimatedCard>
        )}
        
        <AnimatedCard index={2}>
          <DashboardCard icon="shield-checkmark-outline" title="Health Status">
            <Text style={styles.cardMainText}>
                {(userData.predictedConditions || []).length > 0 ? userData.predictedConditions.join(', ') : 'Healthy'}
            </Text>
          </DashboardCard>
        </AnimatedCard>
        
        <AnimatedCard index={3}>
          <DashboardCard icon="list-outline" title="Today's Care Plan">
            {(userData.carePlan || []).map((item: CarePlanItem) => (
              <TouchableOpacity 
                key={item.id} 
                style={styles.carePlanItem} 
                onPress={() => toggleCarePlanItem && toggleCarePlanItem(item.id)}
                activeOpacity={0.6}
              >
                <Ionicons 
                    name={item.completed ? "checkmark-circle" : "ellipse-outline"} 
                    size={28} 
                    color={item.completed ? Colors.accent : Colors.primary}
                />
                <Text 
                    style={[
                        styles.carePlanText, 
                        {
                          textDecorationLine: item.completed ? 'line-through' : 'none',
                          color: item.completed ? Colors.inactive : Colors.text
                        }
                    ]}
                >
                    {item.task}
                </Text>
              </TouchableOpacity>
            ))}
          </DashboardCard>
        </AnimatedCard>

        <AnimatedCard index={4}>
            <DashboardCard icon="pulse-outline" title="Latest Vitals">
            {latestVitals ? (
                <View>
                <Text style={styles.cardSubText}>Recorded on: {new Date(latestVitals.date).toLocaleString()}</Text>
                {latestVitals.glucose && <Text style={styles.cardMainText}>Glucose: {latestVitals.glucose} mg/dL</Text>}
                {latestVitals.bloodPressure && <Text style={styles.cardMainText}>Blood Pressure: {latestVitals.bloodPressure} mmHg</Text>}
                {latestVitals.heartRate && <Text style={styles.cardMainText}>Heart Rate: {latestVitals.heartRate} bpm</Text>}
                </View>
            ) : (
                <Text style={styles.cardSubText}>No readings recorded yet.</Text>
            )}
            </DashboardCard>
        </AnimatedCard>

        <AnimatedCard index={5}>
            <DashboardCard icon="time-outline" title="Reading History">
            <FlatList
                data={userData.vitalHistory || []}
                renderItem={renderHistoryItem}
                keyExtractor={(item, index) => `${item.date}-${index}`}
                ListEmptyComponent={<Text style={styles.cardSubText}>Your history of readings will appear here.</Text>}
                ItemSeparatorComponent={() => <View style={styles.separator} />}
                scrollEnabled={false}
            />
            </DashboardCard>
        </AnimatedCard>
        
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background, },
  centerContent: { justifyContent: 'center', alignItems: 'center' },
  contentContainer: { padding: 20 },
  headerTitle: { fontSize: 28, fontWeight: 'bold', color: Colors.text, },
  headerSubtitle: { fontSize: 16, color: Colors.textSecondary, marginBottom: 20 },
  addButton: { backgroundColor: Colors.primary, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', padding: 15, borderRadius: 15 },
  addButtonText: { color: '#FFFFFF', fontSize: 18, fontWeight: 'bold', marginLeft: 10 },
  cardMainText: { fontSize: 18, fontWeight: '500', color: Colors.text, marginBottom: 5, lineHeight: 24 },
  cardSubText: { fontSize: 14, color: Colors.textSecondary, textAlign: 'center', padding: 10 },
  carePlanItem: { flexDirection: 'row', alignItems: 'center', paddingVertical: 12 },
  carePlanText: { fontSize: 16, marginLeft: 15 },
  historyItem: { paddingVertical: 10, },
  historyHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 8, },
  historyDate: { fontSize: 16, fontWeight: '600', color: '#333', marginLeft: 8, },
  historyVitals: { paddingLeft: 15, borderLeftWidth: 2, borderLeftColor: '#E0E0E0', },
  historyText: { fontSize: 15, color: '#555', lineHeight: 22, },
  historyNotes: { fontSize: 14, color: '#777', fontStyle: 'italic', marginTop: 4, },
  separator: { height: 1, backgroundColor: '#F0F4F8', marginVertical: 5, },
  drilldownButton: { backgroundColor: '#5D6D7E', padding: 15, borderRadius: 10, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', },
  drilldownButtonText: { color: '#FFFFFF', fontSize: 16, fontWeight: '600', },
});