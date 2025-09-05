import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, StatusBar } from 'react-native';
import { Stack } from 'expo-router';
import { Colors } from '../constants/Colors'; // CORRECTED PATH
import { Ionicons } from '@expo/vector-icons';

// --- TYPE DEFINITIONS TO FIX ERRORS ---
type StatCardProps = {
    icon: keyof typeof Ionicons.glyphMap;
    label: string;
    value: string | number;
    unit: string;
    color: string;
};

type RemedyItemProps = {
    icon: keyof typeof Ionicons.glyphMap;
    type: string;
    action: string;
    priority: 'high' | 'medium';
};

type WearableDataState = {
    glucose: number;
    heartRate: number;
    systolicBP: number;
    diastolicBP: number;
    steps: number;
    hrv: number;
    sleepHours: number;
    stressLevel: number;
};
// --- END OF FIXES ---

const StatCard = ({ icon, label, value, unit, color }: StatCardProps) => (
    <View style={styles.statCard}>
        <View style={{flexDirection: 'row', alignItems: 'center', marginBottom: 8}}>
            <Ionicons name={icon} size={20} color={color} />
            <Text style={[styles.statLabel, {color}]}>{label}</Text>
        </View>
        <Text style={[styles.statValue, {color}]}>{value}</Text>
        <Text style={[styles.statUnit, {color: Colors.textSecondary}]}>{unit}</Text>
    </View>
);

const RemedyItem = ({ icon, type, action, priority }: RemedyItemProps) => (
    <View style={[styles.remedyItem, { borderLeftColor: priority === 'high' ? Colors.danger : Colors.primary }]}>
        <View style={styles.remedyHeader}>
            <Ionicons name={icon} size={20} color={Colors.primary} />
            <Text style={styles.remedyType}>{type}</Text>
        </View>
        <Text style={styles.remedyAction}>{action}</Text>
    </View>
);

export default function LiveMonitorScreen() {
    const [isConnected, setIsConnected] = useState(false);
    const [lastUpdate, setLastUpdate] = useState(new Date());
    const [wearableData, setWearableData] = useState<WearableDataState>({
        glucose: 95, heartRate: 72, systolicBP: 120, diastolicBP: 80,
        steps: 8542, hrv: 35, sleepHours: 7.2, stressLevel: 3
    });

    useEffect(() => {
        const interval = setInterval(() => {
            if (isConnected) {
                setWearableData((prev: WearableDataState) => ({
                    glucose: Math.max(70, Math.min(200, prev.glucose + (Math.random() - 0.5) * 10)),
                    heartRate: Math.max(60, Math.min(120, prev.heartRate + (Math.random() - 0.5) * 8)),
                    systolicBP: Math.max(90, Math.min(180, prev.systolicBP + (Math.random() - 0.5) * 6)),
                    diastolicBP: Math.max(60, Math.min(120, prev.diastolicBP + (Math.random() - 0.5) * 4)),
                    steps: prev.steps + Math.floor(Math.random() * 50),
                    hrv: Math.max(15, Math.min(60, prev.hrv + (Math.random() - 0.5) * 3)),
                    stressLevel: Math.max(1, Math.min(10, prev.stressLevel + (Math.random() - 0.5) * 1)),
                    sleepHours: prev.sleepHours
                }));
                setLastUpdate(new Date());
            }
        }, 2000);
        return () => clearInterval(interval);
    }, [isConnected]);

    const getRemedyRecommendations = () => {
        const remedies = [];
        if (wearableData.glucose > 140) remedies.push({ type: 'Insulin', action: 'Consider rapid-acting insulin if glucose >180mg/dL', priority: 'high' as const, icon: 'eyedrop-outline' as const });
        if (wearableData.steps < 5000) remedies.push({ type: 'Activity', action: 'Take a 10-minute walk to improve glucose uptake', priority: 'medium' as const, icon: 'walk-outline' as const });
        if (wearableData.stressLevel > 6) remedies.push({ type: 'Stress', action: 'Practice deep breathing for 5 minutes', priority: 'medium' as const, icon: 'leaf-outline' as const });
        if (wearableData.systolicBP > 140) remedies.push({ type: 'Medication', action: 'Monitor BP closely, consider antihypertensive', priority: 'high' as const, icon: 'heart-outline' as const });
        return remedies;
    };

    const remedies = getRemedyRecommendations();

    return (
        <ScrollView style={styles.container}>
            <Stack.Screen options={{ title: 'Live Monitor' }} />
            <StatusBar barStyle="dark-content" />

            <View style={styles.header}>
                <View>
                    <Text style={styles.headerTitle}>Diabetes Risk Monitor</Text>
                    <Text style={styles.headerSubtitle}>Real-time wearable health insights</Text>
                </View>
                <TouchableOpacity onPress={() => setIsConnected(!isConnected)} style={[styles.connectButton, isConnected ? styles.connected : styles.disconnected]}>
                    <Ionicons name="watch-outline" size={20} color={isConnected ? Colors.accent : Colors.textSecondary} />
                    <Text style={[styles.connectButtonText, isConnected ? {color: Colors.accent} : {color: Colors.textSecondary}]}>{isConnected ? 'Connected' : 'Connect'}</Text>
                </TouchableOpacity>
            </View>

            <View style={styles.card}>
                <View style={styles.cardHeader}>
                    <Text style={styles.cardTitle}>Live Wearable Data</Text>
                    <Text style={styles.updateText}>Last update: {lastUpdate.toLocaleTimeString()}</Text>
                </View>
                <View style={styles.statsGrid}>
                    <StatCard icon="water-outline" label="Glucose" value={wearableData.glucose.toFixed(0)} unit="mg/dL" color={wearableData.glucose > 180 ? Colors.danger : wearableData.glucose > 140 ? '#F39C12' : Colors.accent} />
                    <StatCard icon="heart-outline" label="Heart Rate" value={wearableData.heartRate.toFixed(0)} unit="bpm" color={Colors.primary} />
                    <StatCard icon="pulse-outline" label="Blood Pressure" value={`${wearableData.systolicBP.toFixed(0)}/${wearableData.diastolicBP.toFixed(0)}`} unit="mmHg" color={Colors.primary} />
                    <StatCard icon="walk-outline" label="Steps" value={wearableData.steps.toLocaleString()} unit="today" color={Colors.accent} />
                </View>
            </View>

            <View style={styles.card}>
                <View style={styles.cardHeader}>
                    <Text style={styles.cardTitle}>Smart Recommendations</Text>
                </View>
                {remedies.length > 0 ? (
                    remedies.map((remedy) => <RemedyItem key={remedy.type} {...remedy} />)
                ) : (
                    <View style={styles.allGoodContainer}>
                        <Ionicons name="checkmark-circle" size={24} color={Colors.accent} />
                        <Text style={styles.allGoodText}>All vitals look good!</Text>
                    </View>
                )}
            </View>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.background, },
    header: { padding: 20, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
    headerTitle: { fontSize: 24, fontWeight: 'bold', color: Colors.text },
    headerSubtitle: { fontSize: 14, color: Colors.textSecondary },
    connectButton: { flexDirection: 'row', alignItems: 'center', paddingVertical: 8, paddingHorizontal: 12, borderRadius: 10 },
    connected: { backgroundColor: '#E8F5E9' },
    disconnected: { backgroundColor: '#F5F5F5' },
    connectButtonText: { fontWeight: '600', marginLeft: 8 },
    card: { backgroundColor: Colors.surface, marginHorizontal: 15, marginBottom: 15, borderRadius: 15, padding: 15, shadowColor: "#000", shadowOffset: { width: 0, height: 2, }, shadowOpacity: 0.1, shadowRadius: 8, elevation: 5, },
    cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15, },
    cardTitle: { fontSize: 18, fontWeight: 'bold', color: Colors.text },
    updateText: { fontSize: 12, color: Colors.textSecondary },
    statsGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between' },
    statCard: { width: '48%', backgroundColor: '#F8F9FA', padding: 15, borderRadius: 10, marginBottom: 10, },
    statLabel: { fontSize: 14, fontWeight: '500', marginLeft: 5, },
    statValue: { fontSize: 24, fontWeight: 'bold', marginTop: 5 },
    statUnit: { fontSize: 12, },
    remedyItem: { backgroundColor: '#F8F9FA', padding: 15, borderRadius: 10, marginBottom: 10, borderLeftWidth: 4, },
    remedyHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 5, },
    remedyType: { fontSize: 16, fontWeight: 'bold', color: Colors.text, marginLeft: 10, },
    remedyAction: { fontSize: 14, color: Colors.textSecondary, },
    allGoodContainer: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#E8F5E9', padding: 15, borderRadius: 10, },
    allGoodText: { fontSize: 16, fontWeight: '600', color: Colors.accent, marginLeft: 10, },
});